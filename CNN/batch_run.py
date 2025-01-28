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

# %% [markdown]
# ### Regular Models

# %%
# A simple CNN Model with 3 Conv1d layers and 2 fully connected layers
# Input (1-Dimension 4-Channels)

## Notes:
#   - Bad fully connected size
#   - Too much VRAM
class SimplestCNNClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 8, kernel_size=4)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(8, 32, kernel_size=4)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.padding3 = nn.CircularPad1d((1,2))
        self.conv3 = nn.Conv1d(32, 128, kernel_size=4)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(225)

        self.act4 = nn.ReLU()

        self.linear1 = nn.Linear(28800, 28800*2)
        self.linear2 = nn.Linear(28800*2, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x))
        x = self.adAvgPool2(x)
        
        x = self.conv3(self.padding3(x))
        x = self.adAvgPool3(x)
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

# %%
# A test of a CNN Model with 3 different Conv1d layers concatenated and 2 fully connected layers
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class SimpleCNNClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(SimpleCNNClassifier, self).__init__()

        self.padding = nn.CircularPad1d((1,2))
        
        self.conv1 = nn.Conv1d(4, 4, kernel_size=4, groups=4)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(4, 4, kernel_size=4)        
        self.act2 = nn.ReLU()
        
        self.act3 = nn.ReLU()        
        self.conv3 = nn.Conv1d(4, 8, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        
        self.act4 = nn.ReLU()

        self.linear1 = nn.Linear(14400, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)
        a = self.conv1(self.padding(x))
        a = self.act1(a)        
        
        b = self.conv2(self.padding(x))
        b = self.act2(b)
        
        c = self.conv3(x)
        c = self.act3(c)
        
        x = torch.flatten(torch.cat([a,b,c], dim=1), 1)
        
        x = self.linear1(x)
        
        x = self.act4(x)
        
        x = self.linear2(x)

        return x

# %%
# A test of a CNN Model with 3 different Conv1d layers concatenated, 2 fully connected layers, and dropouts
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class SimpleCNNWithDropoutClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(SimpleCNNWithDropoutClassifier, self).__init__()

        self.padding = nn.CircularPad1d((1,2))

        self.dropout1 = nn.Dropout(p=0.2)
        self.conv1 = nn.Conv1d(4, 4, kernel_size=4, groups=4)
        self.act1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(4, 4, kernel_size=4)
        self.act2 = nn.ReLU()
        
        self.act3 = nn.ReLU()
        self.conv3 = nn.Conv1d(4, 8, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")

        self.act4 = nn.ReLU()

        self.dropout2 = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(14400, 7200)

        self.dropout3 = nn.Dropout(p=0.2)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)
        x = self.dropout1(x)
        a = self.conv1(self.padding(x))
        a = self.act1(a)
        
        b = self.conv2(self.padding(x))
        b = self.act2(b)
        
        c = self.conv3(x)
        c = self.act3(c)
        
        x = torch.flatten(torch.cat([a,b,c], dim=1), 1)
        
        x = self.dropout2(x)
        x = self.linear1(x)
        x = self.act4(x)
        
        x = self.dropout3(x)
        x = self.linear2(x)

        return x

# %%
# A CNN Model with 2 different Conv1d layers concatenated, 2 fully connected layers
# Input (1-Dimension 4-Channels)

## Notes:
#   - Miss sequential Conv layers
class BaseCNNClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(BaseCNNClassifier, self).__init__()

        self.conv1_1 = nn.Conv1d(1, 4, kernel_size=4)
        self.conv1_2 = nn.Conv1d(1, 4, kernel_size=4, dilation=2)
        self.avgPool = nn.AvgPool1d(4, stride=2)
        
        self.padding = nn.CircularPad1d((1,2))
        
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()

        self.linear1 = nn.Linear(14392, 14392)
        self.linear2 = nn.Linear(14392, nClasses)
    
    def forward(self, x):

        x = torch.unsqueeze(torch.flatten(x, start_dim=1), 1)
        x = self.padding(x)
        
        x_1_1 = self.conv1_1(x)
        x_1_2 = self.conv1_2(self.padding(x))      

        x = torch.cat([x_1_1, x_1_2], dim=1)
        x = self.avgPool(x)

        x = torch.flatten(x, 1)
        
        x = self.act1(x)
        x = self.linear1(x)
        
        x = self.act2(x)
        x = self.linear2(x)
        
        return x

# %%
# A UNet Model with 2 encode+decode levels and 2 fully connected layers
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class UnetBasedCNNClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(UnetBasedCNNClassifier, self).__init__()

        # First Encode Level
        self.padding_e_1_1 = nn.CircularPad1d((1,2))
        self.conv_e_1_1 = nn.Conv1d(4, 8, kernel_size=4)
        self.act_e_1_1 = nn.ReLU()

        self.padding_e_1_2 = nn.CircularPad1d((1,2))
        self.conv_e_1_2 = nn.Conv1d(8, 8, kernel_size=4)
        self.act_e_1_2 = nn.ReLU()

        self.avgPool_e_1_1 = nn.AvgPool1d(2, stride=2)


        # Second Encode Level        
        self.padding_e_2_1 = nn.CircularPad1d((1,2))
        self.conv_e_2_1 = nn.Conv1d(8, 16, kernel_size=4)
        self.act_e_2_1 = nn.ReLU()
        
        self.padding_e_2_2 = nn.CircularPad1d((1,2))
        self.conv_e_2_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_e_2_2 = nn.ReLU()

        self.avgPool_e_2_1 = nn.AvgPool1d(2, stride=2)

        
        # Transition Level
        self.padding_t_1_1 = nn.CircularPad1d((1,2))
        self.conv_t_1_1 = nn.Conv1d(16, 32, kernel_size = 4)
        self.act_t_1_1 = nn.ReLU()
        
        self.padding_t_1_2 = nn.CircularPad1d((1,2))
        self.conv_t_1_2 = nn.Conv1d(32, 32, kernel_size = 4)
        self.act_t_1_2 = nn.ReLU()


        # First Decode Level
        self.upconv_1_1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)

        self.padding_d_1_1 = nn.CircularPad1d((1,2))
        self.conv_d_1_1 = nn.Conv1d(32, 16, kernel_size=4)
        self.act_d_1_1 = nn.ReLU()

        self.padding_d_1_2 = nn.CircularPad1d((1,2))
        self.conv_d_1_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_d_1_2 = nn.ReLU()


        # Second Decode Level
        self.upconv_2_1 = nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2)

        self.padding_d_2_1 = nn.CircularPad1d((1,2))
        self.conv_d_2_1 = nn.Conv1d(16, 8, kernel_size=4)
        self.act_d_2_1 = nn.ReLU()

        self.padding_d_2_2 = nn.CircularPad1d((1,2))
        self.conv_d_2_2 = nn.Conv1d(8, 8, kernel_size=4)
        self.act_d_2_2 = nn.ReLU()


        # Output Level
        self.conv_out_1 = nn.Conv1d(8, nClasses, kernel_size=1)
        
        self.linear_out_1 = nn.Linear(7200, 14400)
        self.act_out_1 = nn.ReLU()
        self.linear_out_2 = nn.Linear(14400, nClasses)
        self.act_out_2 = nn.ReLU()
    
    def forward(self, x):

        x = x.reshape(x.shape[0],4,-1)              # Reshape X shape: torch.Size([3, 4, 900])

        # Encoder 1
        x_e_1 = self.padding_e_1_1(x)               # Padding X shape: torch.Size([3, 4, 900])
        x_e_1 = self.conv_e_1_1(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_1(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_1 = self.padding_e_1_2(x_e_1)           # Padding X shape: torch.Size([3, 8, 903])
        x_e_1 = self.conv_e_1_2(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_2(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_p_1 = self.avgPool_e_1_1(x_e_1)         # Pool X shape: torch.Size([3, 8, 450])


        # Encoder 2
        x_e_2 = self.padding_e_2_1(x_e_p_1)         # Padding X shape: torch.Size([3, 8, 453])
        x_e_2 = self.conv_e_2_1(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_1(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])
        
        x_e_2 = self.padding_e_2_2(x_e_2)           # Padding X shape: torch.Size([3, 16, 453])
        x_e_2 = self.conv_e_2_2(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_2(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_p_2 = self.avgPool_e_2_1(x_e_2)         # Pool X shape: torch.Size([3, 16, 225])


        # Transition
        x_t_1 = self.padding_t_1_1(x_e_p_2)         # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_1(x_t_1)              # Conv X shape: torch.Size([3, 32, 109])
        x_t_1 = self.act_t_1_1(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 109])
        
        x_t_1 = self.padding_t_1_2(x_t_1)           # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_2(x_t_1)              # Conv X shape: torch.Size([3, 32, 106])
        x_t_1 = self.act_t_1_2(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 106])


        # Decode 1
        x_d_1 = self.upconv_1_1(x_t_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_1 = torch.cat([x_d_1, x_e_2], dim=1)    # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_1(x_d_1)              # 
        x_d_1 = self.act_d_1_1(x_d_1)               # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_2(x_d_1)              # 
        x_d_1 = self.act_d_1_2(x_d_1)               # 


        # Decode 2
        x_d_2 = self.upconv_2_1(x_d_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_2 = torch.cat([x_d_2, x_e_1], dim=1)    # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_1(x_d_2)              # 
        x_d_2 = self.act_d_2_1(x_d_2)               # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_2(x_d_2)              # 
        x_d_2 = self.act_d_2_2(x_d_2)               # 


        # Output

        x = torch.flatten(x_d_2, 1)
        
        x = self.act_out_1(x)
        x = self.linear_out_1(x)
        
        x = self.act_out_2(x)
        x = self.linear_out_2(x)

        return x

# %%
# A UNet Model variant with 2 encode+decode levels, 2 fully connected layers, and dropouts
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class UnetBasedCNNWithDropoutClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(UnetBasedCNNWithDropoutClassifier, self).__init__()

        self.input_dropout1 = nn.Dropout(p=0.2)

        # First Encode Level
        self.padding_e_1_1 = nn.CircularPad1d((1,2))
        self.conv_e_1_1 = nn.Conv1d(4, 8, kernel_size=4)
        self.act_e_1_1 = nn.ReLU()

        self.padding_e_1_2 = nn.CircularPad1d((1,2))
        self.conv_e_1_2 = nn.Conv1d(8, 8, kernel_size=4)
        self.act_e_1_2 = nn.ReLU()

        self.avgPool_e_1_1 = nn.AvgPool1d(2, stride=2)


        # Second Encode Level        
        self.padding_e_2_1 = nn.CircularPad1d((1,2))
        self.conv_e_2_1 = nn.Conv1d(8, 16, kernel_size=4)
        self.act_e_2_1 = nn.ReLU()
        
        self.padding_e_2_2 = nn.CircularPad1d((1,2))
        self.conv_e_2_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_e_2_2 = nn.ReLU()

        self.avgPool_e_2_1 = nn.AvgPool1d(2, stride=2)

        
        # Transition Level
        self.padding_t_1_1 = nn.CircularPad1d((1,2))
        self.conv_t_1_1 = nn.Conv1d(16, 32, kernel_size = 4)
        self.act_t_1_1 = nn.ReLU()
        
        self.padding_t_1_2 = nn.CircularPad1d((1,2))
        self.conv_t_1_2 = nn.Conv1d(32, 32, kernel_size = 4)
        self.act_t_1_2 = nn.ReLU()


        # First Decode Level
        self.upconv_1_1 = nn.ConvTranspose1d(32, 16, kernel_size=2, stride=2)

        self.padding_d_1_1 = nn.CircularPad1d((1,2))
        self.conv_d_1_1 = nn.Conv1d(32, 16, kernel_size=4)
        self.act_d_1_1 = nn.ReLU()

        self.padding_d_1_2 = nn.CircularPad1d((1,2))
        self.conv_d_1_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_d_1_2 = nn.ReLU()


        # Second Decode Level
        self.upconv_2_1 = nn.ConvTranspose1d(16, 8, kernel_size=2, stride=2)

        self.padding_d_2_1 = nn.CircularPad1d((1,2))
        self.conv_d_2_1 = nn.Conv1d(16, 8, kernel_size=4)
        self.act_d_2_1 = nn.ReLU()

        self.padding_d_2_2 = nn.CircularPad1d((1,2))
        self.conv_d_2_2 = nn.Conv1d(8, 8, kernel_size=4)
        self.act_d_2_2 = nn.ReLU()


        # Output Level
        self.conv_out_1 = nn.Conv1d(8, nClasses, kernel_size=1)
        
        
        self.output_dropout1 = nn.Dropout(p=0.2)
        self.linear_out_1 = nn.Linear(7200, 14400)
        self.act_out_1 = nn.ReLU()

        
        self.output_dropout2 = nn.Dropout(p=0.2)
        self.linear_out_2 = nn.Linear(14400, nClasses)
        self.act_out_2 = nn.ReLU()
    
    def forward(self, x):

        x = x.reshape(x.shape[0],4,-1)              # Reshape X shape: torch.Size([3, 4, 900])
        x = self.input_dropout1(x)

        x_e_1 = self.padding_e_1_1(x)               # Padding X shape: torch.Size([3, 4, 900])
        x_e_1 = self.conv_e_1_1(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_1(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_1 = self.padding_e_1_2(x_e_1)           # Padding X shape: torch.Size([3, 8, 903])
        x_e_1 = self.conv_e_1_2(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_2(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_p_1 = self.avgPool_e_1_1(x_e_1)         # Pool X shape: torch.Size([3, 8, 450])


        x_e_2 = self.padding_e_2_1(x_e_p_1)         # Padding X shape: torch.Size([3, 8, 453])
        x_e_2 = self.conv_e_2_1(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_1(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])
        
        x_e_2 = self.padding_e_2_2(x_e_2)           # Padding X shape: torch.Size([3, 16, 453])
        x_e_2 = self.conv_e_2_2(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_2(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_p_2 = self.avgPool_e_2_1(x_e_2)         # Pool X shape: torch.Size([3, 16, 225])


        x_t_1 = self.padding_t_1_1(x_e_p_2)         # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_1(x_t_1)              # Conv X shape: torch.Size([3, 32, 109])
        x_t_1 = self.act_t_1_1(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 109])
        
        x_t_1 = self.padding_t_1_2(x_t_1)           # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_2(x_t_1)              # Conv X shape: torch.Size([3, 32, 106])
        x_t_1 = self.act_t_1_2(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 106])


        x_d_1 = self.upconv_1_1(x_t_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_1 = torch.cat([x_d_1, x_e_2], dim=1)    # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_1(x_d_1)              # 
        x_d_1 = self.act_d_1_1(x_d_1)               # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_2(x_d_1)              # 
        x_d_1 = self.act_d_1_2(x_d_1)               # 


        x_d_2 = self.upconv_2_1(x_d_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_2 = torch.cat([x_d_2, x_e_1], dim=1)    # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_1(x_d_2)              # 
        x_d_2 = self.act_d_2_1(x_d_2)               # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_2(x_d_2)              # 
        x_d_2 = self.act_d_2_2(x_d_2)               # 


        x = torch.flatten(x_d_2, 1)
        x = self.output_dropout1(x)
        
        x = self.act_out_1(x)
        x = self.linear_out_1(x)
        
        x = self.output_dropout2(x)
        x = self.act_out_2(x)
        x = self.linear_out_2(x)

        return x

# %%
# A UNet Model with 2 encode(with dilation)+decode levels and 2 fully connected layers
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class UnetBasedCNNWithDilationClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(UnetBasedCNNWithDilationClassifier, self).__init__()

        # First Encode Level
        self.padding_e_1_1 = nn.CircularPad1d((1,2))
        self.conv_e_1_1 = nn.Conv1d(4, 8, kernel_size=4)
        self.act_e_1_1 = nn.ReLU()
        
        self.convd_e_1_1 = nn.Conv1d(4, 8, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        self.actd_e_1_1 = nn.ReLU()

        self.padding_e_1_2 = nn.CircularPad1d((1,2))
        self.conv_e_1_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_e_1_2 = nn.ReLU()

        self.avgPool_e_1_1 = nn.AvgPool1d(2, stride=2)

        # Second Encode Level        
        self.padding_e_2_1 = nn.CircularPad1d((1,2))
        self.conv_e_2_1 = nn.Conv1d(16, 32, kernel_size=4)
        self.act_e_2_1 = nn.ReLU()
        
        self.convd_e_2_1 = nn.Conv1d(16, 32, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        self.actd_e_2_1 = nn.ReLU()
        
        self.padding_e_2_2 = nn.CircularPad1d((1,2))
        self.conv_e_2_2 = nn.Conv1d(64, 64, kernel_size=4)
        self.act_e_2_2 = nn.ReLU()

        self.avgPool_e_2_1 = nn.AvgPool1d(2, stride=2)
        
        # Transition Level
        self.padding_t_1_1 = nn.CircularPad1d((1,2))
        self.conv_t_1_1 = nn.Conv1d(64, 128, kernel_size = 4)
        self.act_t_1_1 = nn.ReLU()
        
        self.padding_t_1_2 = nn.CircularPad1d((1,2))
        self.conv_t_1_2 = nn.Conv1d(128, 128, kernel_size = 4)
        self.act_t_1_2 = nn.ReLU()

        # First Decode Level
        self.upconv_1_1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)

        self.padding_d_1_1 = nn.CircularPad1d((1,2))
        self.conv_d_1_1 = nn.Conv1d(128, 64, kernel_size=4)
        self.act_d_1_1 = nn.ReLU()

        self.padding_d_1_2 = nn.CircularPad1d((1,2))
        self.conv_d_1_2 = nn.Conv1d(64, 64, kernel_size=4)
        self.act_d_1_2 = nn.ReLU()

        # Second Decode Level
        self.upconv_2_1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)

        self.padding_d_2_1 = nn.CircularPad1d((1,2))
        self.conv_d_2_1 = nn.Conv1d(48, 24, kernel_size=4)
        self.act_d_2_1 = nn.ReLU()

        self.padding_d_2_2 = nn.CircularPad1d((1,2))
        self.conv_d_2_2 = nn.Conv1d(24, 24, kernel_size=4)
        self.act_d_2_2 = nn.ReLU()

        # Output Level
        self.conv_out_1 = nn.Conv1d(24, nClasses, kernel_size=1)
        
        self.linear_out_1 = nn.Linear(21600, 7200)
        self.act_out_1 = nn.ReLU()

        self.linear_out_2 = nn.Linear(7200, nClasses)
        self.act_out_2 = nn.ReLU()
    
    def forward(self, x):

        x = x.reshape(x.shape[0],4,-1)              # Reshape X shape: torch.Size([3, 4, 900])

        # Encoder 1
        x_e_1 = self.padding_e_1_1(x)               # Padding X shape: torch.Size([3, 4, 900])
        x_e_1 = self.conv_e_1_1(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])        
        x_e_1 = self.act_e_1_1(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])        

        x_e_1_d = self.convd_e_1_1(x)
        x_e_1_d = self.actd_e_1_1(x_e_1_d)
        
        x_e_1 = torch.cat([x_e_1, x_e_1_d], dim=1)
        
        x_e_1 = self.padding_e_1_2(x_e_1)           # Padding X shape: torch.Size([3, 8, 903])
        x_e_1 = self.conv_e_1_2(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_2(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_p_1 = self.avgPool_e_1_1(x_e_1)         # Pool X shape: torch.Size([3, 8, 450])

        # Encoder 2
        x_e_2 = self.padding_e_2_1(x_e_p_1)         # Padding X shape: torch.Size([3, 8, 453])
        x_e_2 = self.conv_e_2_1(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_1(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_2_d = self.convd_e_2_1(x_e_p_1)
        x_e_2_d = self.actd_e_2_1(x_e_2_d)        
        x_e_2 = torch.cat([x_e_2, x_e_2_d], dim=1)
        
        x_e_2 = self.padding_e_2_2(x_e_2)           # Padding X shape: torch.Size([3, 16, 453])
        x_e_2 = self.conv_e_2_2(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_2(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_p_2 = self.avgPool_e_2_1(x_e_2)         # Pool X shape: torch.Size([3, 16, 225])


        # Transition
        x_t_1 = self.padding_t_1_1(x_e_p_2)         # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_1(x_t_1)              # Conv X shape: torch.Size([3, 32, 109])
        x_t_1 = self.act_t_1_1(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 109])
        
        x_t_1 = self.padding_t_1_2(x_t_1)           # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_2(x_t_1)              # Conv X shape: torch.Size([3, 32, 106])
        x_t_1 = self.act_t_1_2(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 106])


        # Decode 1
        x_d_1 = self.upconv_1_1(x_t_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_1 = torch.cat([x_d_1, x_e_2], dim=1)    # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_1(x_d_1)              # 
        x_d_1 = self.act_d_1_1(x_d_1)               # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_2(x_d_1)              # 
        x_d_1 = self.act_d_1_2(x_d_1)               # 


        # Decode 2
        x_d_2 = self.upconv_2_1(x_d_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_2 = torch.cat([x_d_2, x_e_1], dim=1)    # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_1(x_d_2)              # 
        x_d_2 = self.act_d_2_1(x_d_2)               # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_2(x_d_2)              # 
        x_d_2 = self.act_d_2_2(x_d_2)               # 


        # Output
        x = torch.flatten(x_d_2, 1)

        x = self.act_out_1(x)
        x = self.linear_out_1(x)
        
        x = self.act_out_2(x)
        x = self.linear_out_2(x)

        return x

# %%
# A UNet Model variant with 2 encode(with dilation)+decode levels, 2 fully connected layers, and dropouts
# Input (1-Dimension 4-Channels)

## Notes:
#   - 
class UnetBasedCNNWithDropoutAndDilationClassifier(nn.Module):
    
    def __init__(self, nClasses):
        super(UnetBasedCNNWithDropoutAndDilationClassifier, self).__init__()

        self.input_dropout1 = nn.Dropout(p=0.2)

        # First Encode Level
        self.padding_e_1_1 = nn.CircularPad1d((1,2))
        self.conv_e_1_1 = nn.Conv1d(4, 8, kernel_size=4)
        self.act_e_1_1 = nn.ReLU()
        
        self.convd_e_1_1 = nn.Conv1d(4, 8, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        self.actd_e_1_1 = nn.ReLU()

        self.padding_e_1_2 = nn.CircularPad1d((1,2))
        self.conv_e_1_2 = nn.Conv1d(16, 16, kernel_size=4)
        self.act_e_1_2 = nn.ReLU()

        self.avgPool_e_1_1 = nn.AvgPool1d(2, stride=2)

        # Second Encode Level        
        self.padding_e_2_1 = nn.CircularPad1d((1,2))
        self.conv_e_2_1 = nn.Conv1d(16, 32, kernel_size=4)
        self.act_e_2_1 = nn.ReLU()
        
        self.convd_e_2_1 = nn.Conv1d(16, 32, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        self.actd_e_2_1 = nn.ReLU()
        
        self.padding_e_2_2 = nn.CircularPad1d((1,2))
        self.conv_e_2_2 = nn.Conv1d(64, 64, kernel_size=4)
        self.act_e_2_2 = nn.ReLU()

        self.avgPool_e_2_1 = nn.AvgPool1d(2, stride=2)

        # Transition Level
        self.padding_t_1_1 = nn.CircularPad1d((1,2))
        self.conv_t_1_1 = nn.Conv1d(64, 128, kernel_size = 4)
        self.act_t_1_1 = nn.ReLU()
        
        self.padding_t_1_2 = nn.CircularPad1d((1,2))
        self.conv_t_1_2 = nn.Conv1d(128, 128, kernel_size = 4)
        self.act_t_1_2 = nn.ReLU()

        # First Decode Level
        self.upconv_1_1 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2)

        self.padding_d_1_1 = nn.CircularPad1d((1,2))
        self.conv_d_1_1 = nn.Conv1d(128, 64, kernel_size=4)
        self.act_d_1_1 = nn.ReLU()

        self.padding_d_1_2 = nn.CircularPad1d((1,2))
        self.conv_d_1_2 = nn.Conv1d(64, 64, kernel_size=4)
        self.act_d_1_2 = nn.ReLU()

        # Second Decode Level
        self.upconv_2_1 = nn.ConvTranspose1d(64, 32, kernel_size=2, stride=2)

        self.padding_d_2_1 = nn.CircularPad1d((1,2))
        self.conv_d_2_1 = nn.Conv1d(48, 24, kernel_size=4)
        self.act_d_2_1 = nn.ReLU()

        self.padding_d_2_2 = nn.CircularPad1d((1,2))
        self.conv_d_2_2 = nn.Conv1d(24, 24, kernel_size=4)
        self.act_d_2_2 = nn.ReLU()

        # Output Level
        self.conv_out_1 = nn.Conv1d(24, nClasses, kernel_size=1)
        
        self.output_dropout1 = nn.Dropout(p=0.2)
        self.linear_out_1 = nn.Linear(21600, 43200)
        self.act_out_1 = nn.ReLU()

        self.output_dropout2 = nn.Dropout(p=0.2)
        self.linear_out_2 = nn.Linear(43200, nClasses)
        self.act_out_2 = nn.ReLU()
    
    def forward(self, x):
        x = x.reshape(x.shape[0],4,-1)              # Reshape X shape: torch.Size([3, 4, 900])

        x = self.input_dropout1(x)

        # Encode 1
        x_e_1 = self.padding_e_1_1(x)               # Padding X shape: torch.Size([3, 4, 900])
        x_e_1 = self.conv_e_1_1(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])        
        x_e_1 = self.act_e_1_1(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])        

        x_e_1_d = self.convd_e_1_1(x)
        x_e_1_d = self.actd_e_1_1(x_e_1_d)
        
        x_e_1 = torch.cat([x_e_1, x_e_1_d], dim=1)
        
        x_e_1 = self.padding_e_1_2(x_e_1)           # Padding X shape: torch.Size([3, 8, 903])
        x_e_1 = self.conv_e_1_2(x_e_1)              # Conv X shape: torch.Size([3, 8, 900])
        x_e_1 = self.act_e_1_2(x_e_1)               # ActFunc X shape: torch.Size([3, 8, 900])
        
        x_e_p_1 = self.avgPool_e_1_1(x_e_1)         # Pool X shape: torch.Size([3, 8, 450])

        # Encode 2
        x_e_2 = self.padding_e_2_1(x_e_p_1)         # Padding X shape: torch.Size([3, 8, 453])
        x_e_2 = self.conv_e_2_1(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_1(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_2_d = self.convd_e_2_1(x_e_p_1)
        x_e_2_d = self.actd_e_2_1(x_e_2_d)        
        x_e_2 = torch.cat([x_e_2, x_e_2_d], dim=1)
        
        x_e_2 = self.padding_e_2_2(x_e_2)           # Padding X shape: torch.Size([3, 16, 453])
        x_e_2 = self.conv_e_2_2(x_e_2)              # Conv X shape: torch.Size([3, 16, 450])
        x_e_2 = self.act_e_2_2(x_e_2)               # ActFunc X shape: torch.Size([3, 16, 450])

        x_e_p_2 = self.avgPool_e_2_1(x_e_2)         # Pool X shape: torch.Size([3, 16, 225])

        # Transition
        x_t_1 = self.padding_t_1_1(x_e_p_2)         # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_1(x_t_1)              # Conv X shape: torch.Size([3, 32, 109])
        x_t_1 = self.act_t_1_1(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 109])
        
        x_t_1 = self.padding_t_1_2(x_t_1)           # Padding X shape: torch.Size([3, 8, 453])
        x_t_1 = self.conv_t_1_2(x_t_1)              # Conv X shape: torch.Size([3, 32, 106])
        x_t_1 = self.act_t_1_2(x_t_1)               # ActFunc X shape: torch.Size([3, 32, 106])

        # Decode 1
        x_d_1 = self.upconv_1_1(x_t_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_1 = torch.cat([x_d_1, x_e_2], dim=1)    # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_1(x_d_1)              # 
        x_d_1 = self.act_d_1_1(x_d_1)               # 

        x_d_1 = self.padding_e_1_1(x_d_1)           # 
        x_d_1 = self.conv_d_1_2(x_d_1)              # 
        x_d_1 = self.act_d_1_2(x_d_1)               # 

        # Decode 2
        x_d_2 = self.upconv_2_1(x_d_1)              # UpConv X shape: torch.Size([3, 16, 214])

        x_d_2 = torch.cat([x_d_2, x_e_1], dim=1)    # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_1(x_d_2)              # 
        x_d_2 = self.act_d_2_1(x_d_2)               # 

        x_d_2 = self.padding_e_1_1(x_d_2)           # 
        x_d_2 = self.conv_d_2_2(x_d_2)              # 
        x_d_2 = self.act_d_2_2(x_d_2)               # 

        # Output
        x = torch.flatten(x_d_2, 1)

        x = self.output_dropout1(x)
        x = self.act_out_1(x)
        x = self.linear_out_1(x)
        
        x = self.output_dropout2(x)
        x = self.act_out_2(x)
        x = self.linear_out_2(x)

        return x

# %% [markdown]
# ### Temporary models tests

# %%
class SimplestCNNClassifier0(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 8, kernel_size=4)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(8, 32, kernel_size=4)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.padding3 = nn.CircularPad1d((1,2))
        self.conv3 = nn.Conv1d(32, 128, kernel_size=4)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(225)

        self.act4 = nn.ReLU()

        self.linear1 = nn.Linear(28800, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x))
        x = self.adAvgPool2(x)
        
        x = self.conv3(self.padding3(x))
        x = self.adAvgPool3(x)
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layer(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layer, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(28832, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layerk2(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layerk2, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 32, kernel_size=2)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(28864, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layerk4(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layerk4, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 32, kernel_size=4)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(28800, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layer16(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layer16, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 16, kernel_size=3)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(14416, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layerGELU(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layerGELU, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)
        
        self.act1 = nn.GELU()

        self.linear1 = nn.Linear(28832, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layer64c(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layer64c, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 64, kernel_size=3)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(57664, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layerPooling(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layerPooling, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(14400, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier0_1layer64cPooling(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier0_1layer64cPooling, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 64, kernel_size=3)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.act1 = nn.ReLU()

        self.linear1 = nn.Linear(28800, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)
      
        x = torch.flatten(x, 1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier_2layers(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_2layers, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 16, kernel_size=4)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.act4 = nn.ReLU()

        self.linear1 = nn.Linear(7200, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x))
        x = self.adAvgPool2(x)
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier_2layers_concat(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_2layers_concat, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 16, kernel_size=4)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(16, 32, kernel_size=4)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.act4 = nn.ReLU()

        self.linear1 = nn.Linear(14400, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x1 = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x1))
        x2 = self.adAvgPool2(x)
        
        # x = torch.flatten(x, 1)
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        x = torch.cat([x1,x2], dim=1)

        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

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
class ResidualBlockGELU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4):
        super(ResidualBlockGELU, self).__init__()
        
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
        self.act = nn.GELU()
    
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
        out = self.act(out)
        
        return out

class SimplestCNNClassifier_2layers_ResidualGELU(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_2layers_ResidualGELU, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlockGELU(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlockGELU(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Activation and fully connected layers
        self.act = nn.GELU()
        
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
    
class SimplestCNNClassifier_GELU2layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_GELU2layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Activation and fully connected layers
        self.act = nn.GELU()
        
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
class SimplestCNNClassifier_3layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_3layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(14400, 7200)
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
        
        # Third residual block
        x = self.residual_block3(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_4layers_Residual_Pooling(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_4layers_Residual_Pooling, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Two additional residual blocks
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(56)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(7168, 7168)
        self.linear2 = nn.Linear(7168, nClasses)
    
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
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_GELU_4layers_Residual_Pooling(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_GELU_4layers_Residual_Pooling, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Two additional residual blocks
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(56)
        
        # Activation and fully connected layers
        self.act = nn.GELU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(7168, 7168)
        self.linear2 = nn.Linear(7168, nClasses)
    
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
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_4layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_4layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        # self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        # self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        # Two additional residual blocks
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(225)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(28800, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):
        # Move channel dimension
        x = torch.movedim(x, -1, -2)
        
        # First residual block
        x = self.residual_block1(x)
        # x = self.adAvgPool1(x)
        
        # Second residual block
        x = self.residual_block2(x)
        # x = self.adAvgPool2(x)
        
        # Third residual block
        x = self.residual_block3(x)
        x = self.adAvgPool3(x)
        
        # Fourth residual block
        x = self.residual_block4(x)
        x = self.adAvgPool4(x)
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_6layers_Residual(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_6layers_Residual, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(56)
        
        # Two additional residual blocks
        self.residual_block5 = ResidualBlock(128, 256)
        self.adAvgPool5 = nn.AdaptiveAvgPool1d(28)
        
        self.residual_block6 = ResidualBlock(256, 512)
        self.adAvgPool6 = nn.AdaptiveAvgPool1d(14)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(7168, 7168)
        self.linear2 = nn.Linear(7168, nClasses)
    
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
        
        # Flatten and fully connected layers
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        
        return x

# %%
class SimplestCNNClassifier_6layers_Residual2(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_6layers_Residual2, self).__init__()
        
        # Residual blocks with adaptive pooling starting from 600 and reducing
        self.residual_block1 = ResidualBlock(4, 16)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(600)
        
        self.residual_block2 = ResidualBlock(16, 32)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(400)
        
        self.residual_block3 = ResidualBlock(32, 64)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(266)
        
        self.residual_block4 = ResidualBlock(64, 128)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(177)
        
        self.residual_block5 = ResidualBlock(128, 256)
        self.adAvgPool5 = nn.AdaptiveAvgPool1d(118)
        
        self.residual_block6 = ResidualBlock(256, 512)
        self.adAvgPool6 = nn.AdaptiveAvgPool1d(78)
        
        # Activation and fully connected layers
        self.act = nn.ReLU()
        
        # Calculate the input size for linear layers
        # Note: You might need to adjust this based on your specific input dimensions
        self.linear1 = nn.Linear(39936, 7168)
        self.linear2 = nn.Linear(7168, nClasses)
    
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

# %%
class ResidualBlock3k(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ResidualBlock3k, self).__init__()
        
        # Padding to maintain input size
        # With kernel_size 3, we need asymmetric padding
        self.padding = nn.CircularPad1d((1,1))
        
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

class SimplestCNNClassifier_8layers_Residual3k(nn.Module):
    def __init__(self, nClasses):
        super(SimplestCNNClassifier_8layers_Residual3k, self).__init__()
        
        # Residual blocks with adaptive pooling
        self.residual_block1 = ResidualBlock3k(4, 16, kernel_size=3)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)
        
        self.residual_block2 = ResidualBlock3k(16, 32, kernel_size=3)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)
        
        self.residual_block3 = ResidualBlock3k(32, 64, kernel_size=3)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(112)
        
        self.residual_block4 = ResidualBlock3k(64, 128, kernel_size=3)
        self.adAvgPool4 = nn.AdaptiveAvgPool1d(56)
        
        self.residual_block5 = ResidualBlock3k(128, 256, kernel_size=3)
        self.adAvgPool5 = nn.AdaptiveAvgPool1d(28)
        
        self.residual_block6 = ResidualBlock3k(256, 512, kernel_size=3)
        self.adAvgPool6 = nn.AdaptiveAvgPool1d(14)
        
        self.residual_block7 = ResidualBlock3k(512, 1024, kernel_size=3)
        self.adAvgPool7 = nn.AdaptiveAvgPool1d(7)
        
        self.residual_block8 = ResidualBlock3k(1024, 2048, kernel_size=3)
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

# %%
class SimplestCNNClassifier1(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier1, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 8, kernel_size=4)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(8, 32, kernel_size=4)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.padding3 = nn.CircularPad1d((1,2))
        self.conv3 = nn.Conv1d(32, 128, kernel_size=4)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(225)

        self.linear1 = nn.Linear(28800, 14400)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(14400, 7200)
        self.act5 = nn.ReLU()
        self.linear3 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x))
        x = self.adAvgPool2(x)
        
        x = self.conv3(self.padding3(x))
        x = self.adAvgPool3(x)
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)
        x = self.act5(x)
        x = self.linear3(x)

        return x

# %%
class SimplestCNNClassifier2(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier2, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 4), stride=1)
        # Output: (batch_size, 8, 900, 4)
        
        # Second convolutional layer
        self.padding2 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 4), stride=1)
        # Output: (batch_size, 16, 4, 900)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 4), stride=1, padding=(1, 0), padding_mode="circular")
        # Output: (batch_size, 32, 1, 900)
        
        # First fully connected layer
        self.fc1 = nn.Linear((32 * 1 * 900 ), (nClasses*2))
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear((nClasses*2), nClasses)
        
        # Activation function
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x = self.padding1(x)
        x = self.relu(self.conv1(x))

        x = self.padding2(x)
        x = self.relu(self.conv2(x))
        
        x = self.relu(self.conv3(x))

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.relu(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimplestCNNClassifier3(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier3, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 8, kernel_size=4)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(8, 16, kernel_size=4)

        self.padding3 = nn.CircularPad1d((1,2))
        self.conv3 = nn.Conv1d(16, 32, kernel_size=4)

        self.linear1 = nn.Linear(28800, 7200)
        self.act4 = nn.ReLU()
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)
        x = self.conv1(self.padding1(x))
        x = self.conv2(self.padding2(x))        
        x = self.conv3(self.padding3(x))
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier4(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier4, self).__init__()

        self.padding1 = nn.CircularPad1d((1,2))
        self.conv1 = nn.Conv1d(4, 8, kernel_size=3)
        self.adAvgPool1 = nn.AdaptiveAvgPool1d(450)

        self.padding2 = nn.CircularPad1d((1,2))
        self.conv2 = nn.Conv1d(8, 32, kernel_size=3)
        self.adAvgPool2 = nn.AdaptiveAvgPool1d(225)

        self.padding3 = nn.CircularPad1d((1,2))
        self.conv3 = nn.Conv1d(32, 128, kernel_size=3)
        self.adAvgPool3 = nn.AdaptiveAvgPool1d(225)

        self.act4 = nn.GELU()

        self.linear1 = nn.Linear(28800, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        x = self.conv1(self.padding1(x))
        x = self.adAvgPool1(x)

        x = self.conv2(self.padding2(x))
        x = self.adAvgPool2(x)
        
        x = self.conv3(self.padding3(x))
        x = self.adAvgPool3(x)
        
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.act4(x)
        x = self.linear2(x)

        return x

# %%
class SimplestCNNClassifier5(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier5, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 4), stride=1)
        self.adAvgPool1 = nn.AdaptiveAvgPool2d(output_size=(450, 4))
        # Output: (batch_size, 8, 450, 4)
        
        # Second convolutional layer
        self.padding2 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 4), stride=1)
        self.adAvgPool2 = nn.AdaptiveAvgPool2d(output_size=(225, 4))
        # Output: (batch_size, 16, 4, 900)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 4), stride=1, padding=(1, 0), padding_mode="circular")
        # Output: (batch_size, 32, 1, 900)
        
        # First fully connected layer
        self.fc1 = nn.Linear(64800, 7200)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(7200, nClasses)
        
        # Activation function
        self.act = nn.GELU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x1 = self.padding1(x)
        x1 = self.act(self.conv1(x1))
        # print(x1.shape)
        x = self.adAvgPool1(x1)
        # print(x1.shape)

        x2 = self.padding2(x)
        x2 = self.act(self.conv2(x2))
        # print(x2.shape)
        x = self.adAvgPool2(x2)
        # print(x2.shape)
        
        x = self.act(self.conv3(x))
        # print(x.shape)


        # x = torch.cat([x, x1, x2], dim=1)
        # print("cat")
        
        x1 = x1.view(x1.size(0), -1)
        # print(x1.shape)
        x2 = x2.view(x2.size(0), -1)
        # print(x2.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)

        x = torch.cat([x,x1,x2], dim=1)
        # print(x.shape)

        # Flatten
        # x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimplestCNNClassifier5_1layer(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier5_1layer, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 4), stride=1)
        # Output: (batch_size, 8, 450, 4)
        
        # First fully connected layer
        self.fc1 = nn.Linear(115200, 7200)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(7200, nClasses)
        
        # Activation function
        self.act = nn.GELU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x = self.padding1(x)
        x = self.act(self.conv1(x))
        # print(x.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimplestCNNClassifier5_1layer64c(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier5_1layer64c, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 4), stride=1)
        # Output: (batch_size, 8, 450, 4)
        
        # First fully connected layer
        self.fc1 = nn.Linear(230400, 7200)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(7200, nClasses)
        
        # Activation function
        self.act = nn.GELU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x = self.padding1(x)
        x = self.act(self.conv1(x))
        # print(x.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimplestCNNClassifier5_1layerPooling(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier5_1layerPooling, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 4), stride=1)
        self.adAvgPool1 = nn.AdaptiveAvgPool2d(output_size=(450, 4))
        # Output: (batch_size, 8, 450, 4)
        
        # First fully connected layer
        self.fc1 = nn.Linear(57600, 7200)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(7200, nClasses)
        
        # Activation function
        self.act = nn.GELU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x = self.padding1(x)
        x = self.act(self.conv1(x))
        # print(x.shape)
        x = self.adAvgPool1(x)
        # print(x1.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimplestCNNClassifier5_1layerPooling64c(nn.Module):
    
    def __init__(self, nClasses):
        super(SimplestCNNClassifier5_1layerPooling64c, self).__init__()
        
        # First convolutional layer
        # Input: (batch_size, 1, 4, 900)
        self.padding1 = nn.CircularPad2d((1, 2, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 4), stride=1)
        self.adAvgPool1 = nn.AdaptiveAvgPool2d(output_size=(450, 4))
        # Output: (batch_size, 8, 450, 4)
        
        # First fully connected layer
        self.fc1 = nn.Linear(115200, 7200)
        self.dropout1 = nn.Dropout(p=0.2)
        
        # Second fully connected layer
        self.fc2 = nn.Linear(7200, nClasses)
        
        # Activation function
        self.act = nn.GELU()
        
    def forward(self, x):
        x = torch.unsqueeze(x,1)

        # Convolutional layers
        x = self.padding1(x)
        x = self.act(self.conv1(x))
        # print(x.shape)
        x = self.adAvgPool1(x)
        # print(x1.shape)

        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.act(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        
        return x

# %%
class SimpleCNNClassifier1(nn.Module):
    
    def __init__(self, nClasses):
        super(SimpleCNNClassifier1, self).__init__()

        self.padding = nn.CircularPad1d((1,2))
        
        self.conv1 = nn.Conv1d(4, 4, kernel_size=4, groups=4)
        self.act1 = nn.ReLU()
        
        self.conv1_1 = nn.Conv1d(4, 8, kernel_size=4, groups=4, dilation=2, padding=3, padding_mode="circular")
        self.act1_1 = nn.ReLU()        
        
        self.conv2 = nn.Conv1d(12, 24, kernel_size=4)        
        self.act2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(24, 32, kernel_size=4)        
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv1d(32, 64, kernel_size=4)        
        self.act4 = nn.ReLU()        
        
        self.act5 = nn.ReLU()

        self.linear1 = nn.Linear(57600, 7200)
        self.linear2 = nn.Linear(7200, nClasses)
    
    def forward(self, x):

        x = torch.movedim(x, -1, -2)

        a = self.conv1(self.padding(x))
        a = self.act1(a)        
        # print(a.shape)
        
        b = self.conv1_1((x))
        b = self.act1_1(b)
        # print(b.shape)
        
        x = torch.cat([a,b], dim=1)
        # print(x.shape)

        
        x = self.conv2(self.padding(x))
        x = self.act2(x)        
        # print(x.shape)

        x = self.conv3(self.padding(x))
        x = self.act3(x)        
        # print(x.shape)
        
        x = self.conv4(self.padding(x))
        x = self.act4(x)        
        # print(x.shape)
        
        x = torch.flatten(x, 1)
        # print(x.shape)
        
        x = self.linear1(x)
        
        x = self.act5(x)
        
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
def Train_Test(model, loss_fn, optimizer, epochs, learning_rate, batch_size, train_data,test_data,id=""):
    
    print("Model: \t\t\t"+(model._get_name() if not model._get_name() == "OptimizedModule" else model.__dict__["_modules"]["_orig_mod"].__class__.__name__))
    print("  Loss Func.: \t\t"+loss_fn._get_name())
    print("  Optimizer: \t\t"+type(optimizer).__name__)
    print("  Epochs: \t\t"+str(epochs))
    print("  Learning Rate: \t"+str(learning_rate))

    print("\nModel Arch: ")
    print(str(model))
    print("\n\n\n")

    # Test CUDA compatibility
    if torch.cuda.get_device_capability() < (7, 0):
        print("Exiting because torch.compile is not supported on this device.")
        import sys
        sys.exit(0)


    epochs_results = []
    current = {
        "model":(model._get_name() if not model._get_name() == "OptimizedModule" else model.__dict__["_modules"]["_orig_mod"].__class__.__name__),
        "loss_function":loss_fn._get_name(),
        "epoch":None,
        "learning_rate":learning_rate,
        "batch_size":None,
        "train_size":None,
        "test_size":None,
        "optimizer":type(optimizer).__name__,
        "train_acc":None,
        "train_loss":None,
        "test_acc":None,
        "test_loss":None,
    }

    # Prepare batch sizes to use
    if batch_size == "dynamic":
        bss = [15000, 15000, 15000, 1000, 15000, 15000, 15000, 15000, 256, 15000, 15000, 15000, 15000, 128, 15000]
    else:
        bss = [batch_size]
    if len(bss) > epochs:
        bss = bss[0:epochs]
    print("Batch Sizes List: "+str(bss))
    batch_lim = int(epochs/len(bss))
    
    
    t_start = time.time()
    best = {
        "epoch":0,
        "train_acc":0,
        "train_loss":10000000,
        "test_acc":0,
        "test_loss":10000000,
    }
    
    train_loader = None
    test_loader = None
    
    
    scaler = GradScaler(device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First restart
        T_mult=2,  # Period multiplier
        eta_min=1e-10,  # Minimum learning rate
    )

    # Epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")

        # Create DataLoaders with current batch size
        if epoch%batch_lim == 0 and len(bss) > 0:
            if train_loader:
                del train_loader
            if test_loader:
                del test_loader

            batch_size = bss.pop(0)
            train_loader, test_loader = loaders_generator(train_data, test_data, batch_size)

        print("Batch Size: "+str(batch_size))
        
        
        # Train Phase

        model.train()
        train_loss = 0
        train_acc = 0

        # Run train over the batches
        optimizer.zero_grad()
        for batch, (X, y) in enumerate(train_loader):
            with torch.autocast(device_type=device, dtype=torch.float16):
                # Compute prediction and loss
                pred = model(X)
                loss = loss_fn(pred, y)
                
            # Backpropagation
            scaler.scale(loss).backward()

            # Gradien clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update the learning rate
            scheduler.step(epoch + batch / len(train_loader))

            optimizer.zero_grad()

            # Update results
            train_loss += loss.item()
            train_acc += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Train results
        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        print("Last Learning Rate: "+str(scheduler.get_last_lr()[0]))
        print(f"Train Error: \n Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f} \n")


        # Test Phase
        model.eval()
        test_loss = 0
        test_acc = 0

        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                test_acc += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()

        # Test results
        test_loss /= len(test_loader)
        test_acc /= len(test_loader.dataset)
        print(f"Test Error: \n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")


        # Update Results
        if best["test_acc"] < test_acc or (best["test_acc"] == test_acc and best["train_acc"] < train_acc):
            best["epoch"] = epoch+1
            best["test_acc"] = test_acc
            best["test_loss"] = test_loss
            best["train_acc"] = train_acc
            best["train_loss"] = train_loss

            # If accuracy over 50%, export the current best treined model
            if test_acc > 0.5:
                torch.save(model.state_dict(), "/media/stark/Models/Gustavo/"+train_data.level+"/"+str(id)+"_"+current["model"]+".pth")
                                
                    
        current["epoch"] = epoch+1
        current["batch_size"] = batch_size
        current["learning_rate"] = scheduler.get_last_lr()[0]
        current["train_size"] = train_loader.dataset.__len__()
        current["test_size"] = test_loader.dataset.__len__()
        current["train_acc"] = train_acc
        current["train_loss"] = train_loss
        current["test_acc"] = test_acc
        current["test_loss"] = test_loss

        epochs_results.append(current.copy())

    # Save Train/Test iteration information
    pd.DataFrame(epochs_results).to_csv("./results/epochs/"+str(id)+"__"+current["model"]+"_train_test.csv")
    
    print("\n\n")
    print(f"Best Epoch:{best['epoch']} \n\tAccuracy: {(100*best['test_acc']):>0.1f}%, Avg loss: {best['test_loss']:>8f} \n")
    print("Train and Test execution time: "+str(format(time.time()-t_start, '.4f'))+"s")
    print("Done!")

    return best

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

results = []
current = {}

id = 0
time_id = str(int(time.time()))
print("Time ID: "+str(time_id))

splitters = [    
    "prop_0-1/min_5/RandomSplit_0",
    "prop_0-1/min_5/RandomSplit_14",
    "prop_0-1/min_5/RandomSplit_56",
    "prop_0-1/min_5/RandomSplit_84",
    "prop_0-1/min_5/RandomSplit_92",
    "prop_0-1/min_5/RandomSplit_101",
    "prop_0-1/min_5/RandomSplit_105",
    "prop_0-1/min_5/RandomSplit_227",
    "prop_0-1/min_5/StratifiedSplit2_0",
    "prop_0-1/min_5/StratifiedSplit2_14",
    "prop_0-1/min_5/StratifiedSplit2_56",
    "prop_0-1/min_5/StratifiedSplit2_84",
    "prop_0-1/min_5/StratifiedSplit2_92",
    "prop_0-1/min_5/StratifiedSplit2_101",
    "prop_0-1/min_5/StratifiedSplit2_105",
    "prop_0-1/min_5/StratifiedSplit2_227",

    "prop_0-1/min_10/RandomSplit_0",
    "prop_0-1/min_10/RandomSplit_14",
    "prop_0-1/min_10/RandomSplit_56",
    "prop_0-1/min_10/RandomSplit_84",
    "prop_0-1/min_10/RandomSplit_92",
    "prop_0-1/min_10/RandomSplit_101",
    "prop_0-1/min_10/RandomSplit_105",
    "prop_0-1/min_10/RandomSplit_227",    
    "prop_0-1/min_10/StratifiedSplit2_0",
    "prop_0-1/min_10/StratifiedSplit2_14",
    "prop_0-1/min_10/StratifiedSplit2_56",
    "prop_0-1/min_10/StratifiedSplit2_84",
    "prop_0-1/min_10/StratifiedSplit2_92",
    "prop_0-1/min_10/StratifiedSplit2_101",
    "prop_0-1/min_10/StratifiedSplit2_105",
    "prop_0-1/min_10/StratifiedSplit2_227",
    
    "prop_0-2/min_5/RandomSplit_0",
    "prop_0-2/min_5/RandomSplit_14",
    "prop_0-2/min_5/RandomSplit_56",
    "prop_0-2/min_5/RandomSplit_84",
    "prop_0-2/min_5/RandomSplit_92",
    "prop_0-2/min_5/RandomSplit_101",
    "prop_0-2/min_5/RandomSplit_105",
    "prop_0-2/min_5/RandomSplit_227",
    "prop_0-2/min_5/StratifiedSplit2_0",
    "prop_0-2/min_5/StratifiedSplit2_14",
    "prop_0-2/min_5/StratifiedSplit2_56",
    "prop_0-2/min_5/StratifiedSplit2_84",
    "prop_0-2/min_5/StratifiedSplit2_92",
    "prop_0-2/min_5/StratifiedSplit2_101",
    "prop_0-2/min_5/StratifiedSplit2_105",
    "prop_0-2/min_5/StratifiedSplit2_227",
    
    "prop_0-2/min_10/RandomSplit_0",
    "prop_0-2/min_10/RandomSplit_14",
    "prop_0-2/min_10/RandomSplit_56",
    "prop_0-2/min_10/RandomSplit_84",
    "prop_0-2/min_10/RandomSplit_92",
    "prop_0-2/min_10/RandomSplit_101",
    "prop_0-2/min_10/RandomSplit_105",
    "prop_0-2/min_10/RandomSplit_227",
    "prop_0-2/min_10/StratifiedSplit2_0",
    "prop_0-2/min_10/StratifiedSplit2_14",
    "prop_0-2/min_10/StratifiedSplit2_56",
    "prop_0-2/min_10/StratifiedSplit2_84",
    "prop_0-2/min_10/StratifiedSplit2_92",
    "prop_0-2/min_10/StratifiedSplit2_101",
    "prop_0-2/min_10/StratifiedSplit2_105",
    "prop_0-2/min_10/StratifiedSplit2_227",
    
    "prop_0-05/min_10/RandomSplit_0",
    "prop_0-05/min_10/RandomSplit_14",
    "prop_0-05/min_10/RandomSplit_56",
    "prop_0-05/min_10/RandomSplit_84",
    "prop_0-05/min_10/RandomSplit_92",
    "prop_0-05/min_10/RandomSplit_101",
    "prop_0-05/min_10/RandomSplit_105",
    "prop_0-05/min_10/RandomSplit_227",
    "prop_0-05/min_10/StratifiedSplit2_0",
    "prop_0-05/min_10/StratifiedSplit2_14",
    "prop_0-05/min_10/StratifiedSplit2_56",
    "prop_0-05/min_10/StratifiedSplit2_84",
    "prop_0-05/min_10/StratifiedSplit2_92",
    "prop_0-05/min_10/StratifiedSplit2_101",
    "prop_0-05/min_10/StratifiedSplit2_105",
    "prop_0-05/min_10/StratifiedSplit2_227",
]

apply_augmentation = [
    False,
    # True,
]

for mat_mul in [False]:#, True]:
    for augmentation in apply_augmentation:
        for level in levels:
            for splitter in splitters:
                clear()

                # Load train and test datasets
                train_data = pd.read_csv("../new_data/"+splitter+"/"+level+"/train_dataset.csv")#[0:1000]
                test_data = pd.read_csv("../new_data/"+splitter+"/"+level+"/test_dataset.csv")#[0:1000]
                print(level)
                print(train_data.shape)
                print(test_data.shape)

                dataset = SequenceDataset(
                    train=train_data, 
                    test=test_data, 
                    level=level, 
                    augmentation=augmentation)
                
                print(dataset.__len__())
                print(dataset.test.__len__())


                for batch_size in hiperparams["batch_size"]:
                    for epochs in hiperparams["epochs"]:
                        for model in hiperparams["model"]:
                            for loss_function_name, loss_function in hiperparams["loss_function"].items():
                                for learning_rate in hiperparams["learning_rate"]:
                                    for optimizer in hiperparams["optimizer"]:
                                        clear()
                                        
                                        optim = optimizer["optim"]
                                        optim_params = optimizer["params"] if "params" in optimizer.keys() else {}

                                        current = {
                                                "id": id,
                                                "start_time":time.time(),
                                                "end_time": None,
                                                "level": level,
                                                "splitter": splitter,
                                                "augmentation": augmentation,
                                                "batch_size": batch_size,
                                                "epochs": epochs,
                                                "model": model.__name__,
                                                "loss_function": loss_function_name+" ("+str(loss_function["function"])+")",
                                                "learning_rate": learning_rate,
                                                "optimizer": optim.__name__+" (params: "+str(optim_params)+")",
                                                "mat_mul": mat_mul,
                                                "obs": "9:1 _ min:5",
                                                "reserved_memory": None,
                                                "error": None
                                            }


                                        try:
                                            # Change precision to improve model performance 
                                            # if mat_mul:
                                            #     torch.set_float32_matmul_precision('high')
                                            # else:
                                            #     torch.set_float32_matmul_precision('highest')
                                            
                                            # Initialize a compiled model, loss function, and optimizer
                                            _model_ = torch.compile(model(dataset.encoded_labels.shape[1]))
                                            _lossfunction_ = loss_function["function"](**{func:params[0](*params[1:]) for func,params in loss_function["function_params"].items()})
                                            _optimizer_ = optim(_model_.parameters(), lr=learning_rate, **optim_params)


                                            # Runt Train-Test
                                            result = Train_Test(
                                                model=_model_,
                                                loss_fn=_lossfunction_,
                                                optimizer=_optimizer_,
                                                epochs=epochs,
                                                learning_rate=learning_rate,
                                                batch_size=batch_size,
                                                train_data=dataset,
                                                test_data=dataset.get_test(),
                                                id=time_id+"_"+str(id),
                                                )
                                                
                                            current["end_time"] = time.time()
                                            current["best_epoch"] = result["epoch"]
                                            current["train_acc_best_epoch"] = result["train_acc"]
                                            current["train_loss_best_epoch"] = result["train_loss"]
                                            current["test_acc_best_epoch"] = result["test_acc"]
                                            current["test_loss_best_epoch"] = result["test_loss"]

                                            current["reserved_memory"] = torch.cuda.memory_reserved() / 1024 / 1024  # Convert to MB

                                            clear()                                
                                            
                                        except Exception as e:
                                            print(e)
                                            current["error"] = str(e)
                                        
                                        # Save the results
                                        results.append(current)
                                        pd.DataFrame(results).to_csv("./results/summarized/"+str(time_id)+"_models_train_test_"+str(len(results))+".csv")
                                        
                                        id = id+1

clear()

# %%



