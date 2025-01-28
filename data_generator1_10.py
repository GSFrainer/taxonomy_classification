# %% [markdown]
# ## Imports

# %%
import pandas as pd
import numpy as np

import os

from sklearn.model_selection import StratifiedKFold, train_test_split

import torch

dtype = torch.float
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
torch.get_default_device()

# %% [markdown]
# # Data Load

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


# Load and filter the data from csv
def load_data(dataset, level, minimun_entries):
    data = dataset.loc[dataset[level].notna()]
    data = data.loc[data["truncated_sequence"].str.len() >= 900].sample(frac=1, random_state=42)

    # Remove sequences classified in more than one class
    tmp = data.groupby("truncated_sequence")[level].nunique().reset_index()
    tmp = tmp.loc[tmp[level]>1]["truncated_sequence"]
    data = data.loc[~data.truncated_sequence.isin(tmp)]

    # Remove duplicates on current level
    data.drop_duplicates(subset=[level, "truncated_sequence"], inplace=True)

    # Remove entries from classes with lass than "minimun_entries" datapoints
    count_classes = data[level].value_counts().reset_index()
    selected_classes = count_classes.loc[count_classes["count"] >= minimun_entries]
    data = data.loc[data[level].isin(selected_classes[level])]
    
    return data

# %%
# Reference map for IUPAC sequences encode
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

# %%
# Load the base dataset
csv = pd.read_csv("./data/cleaned_sequences.csv", 
                  usecols=[
                      'domain', 
                      'supergroup', 
                      'division', 
                      'subdivision', 
                      'class', 
                      'order', 
                      'family', 
                      'genus', 
                      'species', 
                      'truncated_sequence'
                     ])
csv.head(1)

# %% [markdown]
# # Data Export 

# %%
# Base path to export the generated data
base_path = "./new_data"

# %%
# Taxonomy levels to filter
levels = ["domain", "class", "order", "family", "genus", "species"]

# Format the row to the content format of the taxonomy file
def taxonomy_format(row, target_level):
    tax = []
    for level in levels:
        if level in row.index:
            tax.append(str(level[0])+"__"+("" if pd.isna(row[level]) else row[level]))
            if level == target_level:
                break
    row["taxonomy"] = "; ".join(tax)
    return row

# Export data to a taxonomy file
def taxonomy_generate(df, target_level, name, path):
    tsv = df.apply(taxonomy_format, axis=1, args=(target_level,)).reset_index(names="seq_id")
    tsv[["seq_id", "taxonomy"]].to_csv(path+"/"+name+"_taxonomy.txt", sep="\t", header=False, index=False, )


# %%
# Generate the fasta file with the dataset data
def fasta_generate(df, name, path):
    with open(path+"/"+name+".fasta", "w+") as fasta:
        for index, row in df.iterrows():
            fasta.write(">"+str(index)+"\n")
            fasta.write(row["truncated_sequence"]+"\n")
                
        fasta.close()

# %%
prop = 0.10        # Train size
k_min = 10          # Minimum n of entries per class
k_splits = k_min    # N of clusters for StratifiedSplit with KFold

def StratifiedSplit(data, level, rand=42):
    _, (X, y) = next(enumerate(StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=rand).split(data.index, data[level])))
    return (data.iloc[X], data.iloc[y])

def StratifiedSplit2(data, level, rand=42):
    return train_test_split(data, test_size=prop, stratify=data[level], random_state=rand)

def RandomSplit(data, level=None, rand=42):
    test_data = data.sample(frac=prop, random_state=rand)
    return (data.drop(test_data.index), test_data)

# %%
# Split functions to be executed
splitters = [
    # StratifiedSplit, 
    StratifiedSplit2,
    RandomSplit,
    ]

# Generate and export the files for each of selected level
for target_level in ["class", "order", "family", "genus", "species"]:

    # Load data and filter the classes with at least K entries
    dataset = load_data(csv, target_level, k_min)
    
    #Remove subsequent levels
    for l in levels[levels.index(target_level)+1:]:
        dataset[l] = np.nan
    dataset=dataset.dropna(subset=levels[:levels.index(target_level)])
    
    for randomness in [0, 14, 56, 92, 84, 101, 105, 227]:
        for splitter in splitters:
            path = base_path+"/prop_"+(str(prop).replace(".", "-"))+"/min_"+str(k_splits)+"/"+splitter.__name__+"_"+str(randomness)+"/"+target_level

            train_dataset, test_dataset = splitter(dataset, level=target_level, rand=randomness)

            print("Level: "+target_level)
            print("Split: "+splitter.__name__+"_"+str(randomness))
            print("Train size: "+str(train_dataset.shape[0]))
            print("Test size: "+str(test_dataset.shape[0]))
            print("\n")

            # print("Original Distribution:")
            # print(dataset[target_level].value_counts(normalize=True))
            # print("\nTrain Set Distribution:")
            # print(train_dataset[target_level].value_counts(normalize=True))
            # print("\nTest Set Distribution:")
            # print(test_dataset[target_level].value_counts(normalize=True))
            # print("\n")
            # break

            # Generate files with the current and previous levels
            os.makedirs(path, exist_ok=True)
            train_dataset.to_csv(path+"/train_dataset.csv")
            test_dataset.to_csv(path+"/test_dataset.csv")
            
            taxonomy_generate(train_dataset, target_level, "pr2_train", path)
            fasta_generate(train_dataset, "pr2_train", path)

            taxonomy_generate(test_dataset, target_level, "pr2_test", path)
            fasta_generate(test_dataset, "pr2_test", path)

            # Generate files only with the current level 
            # path = base_path+"/Isolated"+splitter.__name__+"/"+target_level
            # os.makedirs(path, exist_ok=True)
            # train_dataset = train_dataset[[target_level, "truncated_sequence"]]
            # test_dataset = test_dataset[[target_level, "truncated_sequence"]]
            
            # taxonomy_generate(train_dataset, target_level, "pr2_train", path)
            # fasta_generate(train_dataset, "pr2_train", path)

            # taxonomy_generate(test_dataset, target_level, "pr2_test", path)
            # fasta_generate(test_dataset, "pr2_test", path)


# %%


# %% [markdown]
# 


