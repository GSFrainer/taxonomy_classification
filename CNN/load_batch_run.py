# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.get_default_device()

# %% [markdown]
# ## Data Load

# %% [markdown]
# ### Data augmentation functions

# %%
def augmentation_bernoulli(seq, prob=0.005):
    idx = torch.bernoulli(prob * torch.ones(len(seq))).nonzero().squeeze(dim=1)
    s = list(seq)

    for i in idx.tolist():
        s[i] = "N"

    return "".join(s)

def sequences_augmentation(data, level, cat, n):
    to_copy = data.loc[data[level] == cat]

    new_data = to_copy[0:1]
    new_data = new_data.drop(new_data.index[0])

    while new_data.shape[0] < n:
        qnt = ((n-(new_data.shape[0])) / to_copy.shape[0]).__ceil__()

        new_data = pd.concat(([to_copy]*qnt)+[new_data])
        new_data["truncated_sequence"] = new_data["truncated_sequence"].apply(augmentation_bernoulli, prob=0.002)
        new_data = new_data.drop_duplicates(subset=["truncated_sequence"])
    
    new_data = new_data[:n-to_copy.shape[0]]
    return new_data

def data_augmentation(data, level, lower, upper):
    class_count = data.groupby(level)[level].count().reset_index(name="count")
    
    cats = class_count.loc[(class_count["count"] < upper) & (class_count["count"] >= lower)][level].to_list()

    clones = sequences_augmentation(data, level, cats[0], upper)
    for cat in cats[1:]:
        clones = pd.concat([clones, sequences_augmentation(data, level, cat, upper)])

    return pd.concat([data, clones])


# %% [markdown]
# ### Sequence encoder

# %%
base_map = {
    "A":[1.0, 0.0, 0.0, 0.0],
    "T":[0.0, 1.0, 0.0, 0.0],
    "G":[0.0, 0.0, 1.0, 0.0],
    "C":[0.0, 0.0, 0.0, 1.0],

    'W':[0.5, 0.5, 0.0, 0.0],
    'S':[0.0, 0.0, 0.5, 0.5],
    'M':[0.5, 0.0, 0.0, 0.5],
    'K':[0.0, 0.5, 0.5, 0.0],
    'R':[0.5, 0.0, 0.5, 0.0],
    'Y':[0.0, 0.5, 0.0, 0.5],
    
    'B':[0.0, 0.3, 0.3, 0.3],
    'D':[0.3, 0.3, 0.3, 0.0],
    'H':[0.3, 0.3, 0.0, 0.3],
    'V':[0.3, 0.0, 0.3, 0.3],

    'N':[0.25, 0.25, 0.25, 0.25],
}

def encode_sequence(sequence):
    encoded_seq = []

    for base in sequence:
        encoded_seq.append(base_map[base])
    
    return torch.tensor(encoded_seq)

# %% [markdown]
# ### PyTorch dataset object to load Sequences and Classification Data

# %%
class SequenceDataset(Dataset):
    def __init__(self, train, test, level, augmentation=False):

        self.classes = pd.concat([train[level], test[level]]).unique().tolist()
        self.classes.sort()
        self.level = level

        if augmentation:
            train = data_augmentation(train, level, 10, 500)
        
        self.labels = train[level]
        self.encoded_labels = SequenceDataset.__encoded_labels__(self.classes, self.labels)
        self.sequences = SequenceDataset.__sequences__(train)

        self.test = SequenceDatasetTest(
            labels = test[level],
            classes = self.classes,
            encoded_labels = SequenceDataset.__encoded_labels__(self.classes, test[level]),
            sequences = SequenceDataset.__sequences__(test)
            )

    def __encoded_labels__(classes, labels):
        return torch.nn.functional.one_hot(torch.tensor([classes.index(l) for l in labels]), len(classes)).type(torch.cuda.FloatTensor)
    
    def __sequences__(ds):
        sequences = []
        for _, row in ds.iterrows():
            sequences.append(encode_sequence(row["truncated_sequence"]))        
        return torch.stack(sequences, dim=0)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return   self.sequences[idx], self.encoded_labels[idx]
    
    def __getitems__(self, ids):
        idx = torch.tensor(ids, device=torch.device('cuda:0'))
        return   list(zip(torch.index_select(self.sequences, 0, idx), torch.index_select(self.encoded_labels, 0, idx)))
    
    def get_test(self):
        return self.test

class SequenceDatasetTest(SequenceDataset):    
    def __init__(self, labels, classes, encoded_labels, sequences):
        self.labels = labels
        self.classes = classes
        self.encoded_labels = encoded_labels
        self.sequences = sequences

# %% [markdown]
# ### Generate PyTorch DataLoader objects

# %%
def loaders_generator(ds_train, ds_test, bs = 128):
    train_loader = DataLoader(ds_train, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))
    test_loader = DataLoader(ds_test, batch_size=bs, shuffle=True, generator=torch.Generator(device='cuda'))

    return train_loader, test_loader

# %% [markdown]
# ## Models

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super(ResidualBlock, self).__init__()
        
        # Padding to maintain input size
        self.padding = nn.CircularPad1d((1,2))
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Store the input for the residual connection
        residual = x
        
        # Main path
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        
        # Shortcut connection
        residual = self.shortcut(residual)
        
        # Add residual connection
        out += residual
        out = self.relu(out)
        
        return out

class SimplestCNNClassifier_2layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_2layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(7200, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):
        # Move channel dimension
        x = torch.movedim(x, -1, -2)
        
        # First residual block
        x = self.residual_block1(x)
        x = self.adAvgPool1(x)
        
        # Second residual block
        x = self.residual_block2(x)
        x = self.adAvgPool2(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_8layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_8layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(56)
        
        self.residual_block5 = ResidualBlock(128, 256)
        self.adAvgPool5 = nn.AdaptiveAvgPool1d(28)
        
        self.residual_block6 = ResidualBlock(256, 512)
        self.adAvgPool6 = nn.AdaptiveAvgPool1d(14)
        
        # Two additional residual blocks
        self.residual_block7 = ResidualBlock(512, 1024)
        self.adAvgPool7 = nn.AdaptiveAvgPool1d(7)
        
        self.residual_block8 = ResidualBlock(1024, 2048)
        self.adAvgPool8 = nn.AdaptiveAvgPool1d(3)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(6144, 6144)
        self.linear2 = nn.Linear(6144, nClasses)
    
    def forward(self, x):
        # Move channel dimension
        x = torch.movedim(x, -1, -2)
        
        # First residual block
        x = self.residual_block1(x)
        x = self.adAvgPool1(x)
        
        # Second residual block
        x = self.residual_block2(x)
        x = self.adAvgPool2(x)
        
        # Third residual block
        x = self.residual_block3(x)
        x = self.adAvgPool3(x)
        
        # Fourth residual block
        x = self.residual_block4(x)
        x = self.adAvgPool4(x)
        
        # Fifth residual block
        x = self.residual_block5(x)
        x = self.adAvgPool5(x)
        
        # Sixth residual block
        x = self.residual_block6(x)
        x = self.adAvgPool6(x)
        
        # Seventh residual block
        x = self.residual_block7(x)
        x = self.adAvgPool7(x)
        
        # Eighth residual block
        x = self.residual_block8(x)
        x = self.adAvgPool8(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %% [markdown]
# ## Test Params

# %%
levels = [
    "class", 
    "order", 
    "family", 
    "genus",
    "species",
]

# %%
batch_sizes = [
    # 64,
    # 128,
    # 256,
    # 512,
    # 2048,
    # 10000,
    "dynamic"
]

# %%
epochs = [
    # 1,
    # 2,
    # 5,
    # 20,
    # 50,
    # 100,
    # 150,
    # 200,
    # 300,
    # 500,
    # 600,
    700,
    # 1000,
]

# %%
models_list = [

    # SimplestCNNClassifier_6layers_Residual,
    # SimplestCNNClassifier_6layers_Residual2,
    SimplestCNNClassifier_8layers_Residual,
    # SimplestCNNClassifier_8layers_Residual3k,

    # SimplestCNNClassifier_GELU_4layers_Residual_Pooling,
    # SimplestCNNClassifier_4layers_Residual,
    # SimplestCNNClassifier_4layers_Residual_Pooling,
    # SimplestCNNClassifier_2layers_ResidualGELU,
    # SimplestCNNClassifier_GELU2layers_Residual,

    # # SimplestCNNClassifier_2layers,
    # SimplestCNNClassifier_2layers_Residual,
    # # # SimplestCNNClassifier_2layers_concat,
    # SimplestCNNClassifier_3layers_Residual,

    # # # SimplestCNNClassifier0_1layer,
    # # SimplestCNNClassifier0_1layerPooling,
    # # # SimplestCNNClassifier0_1layerGELU,
    # # # SimplestCNNClassifier0_1layer64c,
    # # # SimplestCNNClassifier0_1layer64cPooling,
    # # SimplestCNNClassifier5_1layer,
    # # # # SimplestCNNClassifier5_1layer64c,
    # # # # SimplestCNNClassifier5_1layerPooling,
    # # # # SimplestCNNClassifier5_1layerPooling64c,

    # # # SimplestCNNClassifier0_1layer16,
    # # # SimplestCNNClassifier0_1layerk4,
    # # SimplestCNNClassifier0_1layerk2,

    # # SimplestCNNClassifier0,
    # # # SimplestCNNClassifier1,
    # # # SimplestCNNClassifier2,
    # # # SimplestCNNClassifier3,
    # # SimplestCNNClassifier5,
    # # # SimpleCNNClassifier1,


    # # # SimplestCNNClassifier,
    # # # SimpleCNNClassifier,
    # # # SimpleCNNWithDropoutClassifier,
    # # # BaseCNNClassifier,
    # # # UnetBasedCNNClassifier,
    # # # UnetBasedCNNWithDropoutClassifier,
    # # # UnetBasedCNNWithDilationClassifier,
    # # # UnetBasedCNNWithDropoutAndDilationClassifier,
]

# %%
loss_functions = {
    "CrossEntropyLoss":{
        "function":nn.CrossEntropyLoss,
        "params":{},
        "function_params":{}
    },
}

# %%
learning_rates = [
    # 5e-2,
    # 1e-2,
    5e-3,
    # 1e-3,
    # 5e-4,
    # 1e-4,
]

# %%
optimizers = [
    {
        "optim":torch.optim.AdamW,
        "params":{
            "weight_decay":1.0,
            "amsgrad":True
        }
    },
]

# %%
hiperparams = {
    "batch_size": batch_sizes,
    "epochs": epochs,
    "model": models_list,
    "loss_function": loss_functions,
    "learning_rate": learning_rates,
    "optimizer": optimizers    
}

# %%
hiperparams

# %% [markdown]
# ## Batch Execution

# %% [markdown]
# ### Train and Test function

# %%
experiment_id = "1734322688"
data = pd.read_csv("./results/summarized/"+experiment_id+"_models_train_test_400.csv")
data.head(2)

# %%
import gc

# Global references
_model_ = None
_lossfunction_ = None
_optimizer_ = None

# Function to clean cache
def clear():
    global _model_, _lossfunction_, _optimizer_
    
    torch.cuda.empty_cache()
    torch.compiler.reset()
    torch._dynamo.reset()

    if _model_:
        del _model_
        _model_ = None
    if _lossfunction_:
        del _lossfunction_
        _lossfunction_ = None
    if _optimizer_:
        del _optimizer_
        _optimizer_ = None
    
    torch.cuda.empty_cache()
    gc.collect()

# %%
times = []
i = 0

# levels = ["class"]

for i, row in data.iterrows():
    # if row["level"] in levels:
    #     continue

    clear()
    
    path = "/media/stark/Models/Gustavo/"+row["level"]+"/"+experiment_id+"_"+str(row["id"])+"_"+str(row["model"])+".pth"
    print(path)

    ## Load Dataset
    train_data = pd.read_csv("../new_data/"+row["splitter"]+"/"+row["level"]+"/train_dataset.csv")#[0:1]
    train_data = train_data.groupby(row["level"]).first().reset_index()
    test_data = pd.read_csv("../new_data/"+row["splitter"]+"/"+row["level"]+"/test_dataset.csv")#[0:1000]

    # print(train_data)

    # print(train_data.shape)
    # print(test_data.shape)

    start_time = time.time()

    dataset = SequenceDataset(
        train=train_data, 
        test=test_data, 
        level=row["level"], 
        augmentation=False)


    train_data=dataset,
    test_data=dataset.get_test()

    ## Load Model
    _model_ = torch.compile(SimplestCNNClassifier_8layers_Residual(dataset.encoded_labels.shape[1]))
    _model_.load_state_dict(torch.load(path, weights_only=True))
    _model_.eval()

    ## Test Model    
    test_loader = DataLoader(test_data, batch_size=15000, shuffle=True, generator=torch.Generator(device='cuda'))
    test_loss = 0
    test_acc = 0

    pred_time = time.time()
    with torch.no_grad():
        for X, y in test_loader:
            pred = _model_(X)
            test_acc += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

    test_acc /= len(test_loader.dataset)

    end_time = time.time()
    
    times.append({
        "experiment_id":experiment_id,
        "id":str(row["id"]),
        "level":row["level"],
        "model":str(row["model"]),
        "splitter":row["splitter"],
        "batch_size":15000,
        "reserved_memory": torch.cuda.memory_reserved() / 1024 / 1024,
        "acc": str(test_acc),
        "start_time":start_time,
        "pred_time":pred_time,
        "end_time":end_time,
    })

    print("Test Acc: "+str(test_acc))

    i = i + 1
    pd.DataFrame(times).to_csv("./results/times/"+experiment_id+"_"+str(i)+"_"+row["model"]+"_times.csv")


# %%



