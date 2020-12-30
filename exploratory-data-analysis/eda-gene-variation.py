# EDA - Distribution of Genes and Variations among Classes

### Import Libraries

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.style.use("bmh")
warnings.filterwarnings("ignore")

### set directory
Im_dir_path = "/home/zhendi/pm/scripts/image/EDA/"

### inport data
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")

### utils for saving and loading
def save_img(fig, path, name):
    file_path = os.path.join(path, name)
    fig.savefig(file_path)


def save_obj(obj, file_address):
    with open(file_address, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_address):
    with open(file_address, "rb") as f:
        return pickle.load(f)


#### Univariant Analysis

# Distribution of classes
fig = plt.figure(figsize=(12, 8))
sns.countplot(x="Class", data=result)
plt.ylabel("Frequency", fontsize=15)
plt.xlabel("Class", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Frequency of Classes", fontsize=20)
save_img(fig, Im_dir_path, "DistributionOfClasses.png")
plt.show()

# Distribution of genes
Genes = result.groupby("Gene")["Gene"].count()
top_genes = Genes.sort_values(ascending=False)[:20]
fig = plt.figure(figsize=(12, 8))
plt.hist(Genes.values, bins=100, log=True, color="#195e83")
plt.xlabel("Number of Unique Genes", fontsize=15)
plt.ylabel("Log of Frequency", fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Distribution of Genes", fontsize=20)
save_img(fig, Im_dir_path, "DistributionOfGenes.png")
plt.show()

# Distribution of variations
Variation = result.groupby("Variation")["Variation"].count()
fig = plt.figure(figsize=(12, 8))
plt.hist(Variation.values, bins=100, log=True, color="#e07b39")
plt.xlabel("Number of Unique Variations", fontsize=15)
plt.ylabel("Log of Frequency", fontsize=15)
plt.title("Distribution of Variations", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
save_img(fig, Im_dir_path, "DistributionOfVariations.png")
plt.show()

### Bivariant Analysis

# Interaction between Genes and Classes
gene_counted = (
    result.groupby(["Class", "Gene"])["Gene"].count().reset_index(name="count")
)
gene_sorted = pd.DataFrame()
for i in range(9):
    index = i + 1
    selected = gene_counted[gene_counted["Class"] == index]
    selected = selected.sort_values("count", ascending=False)[:8]
    print(selected)
    gene_sorted = pd.concat([gene_sorted, selected], axis=0)

# Top 8 genes among 9 classes
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(18, 18))
for i in range(3):
    for j in range(3):
        sns.barplot(
            x="Gene",
            y="count",
            data=gene_sorted[gene_sorted["Class"] == ((i * 3 + j) + 1)],
            ax=ax[i][j],
        )
        ax[i, j].set_title("Class %d" % ((i * 3 + j) + 1), fontsize=20)
        ax[i, j].set_xlabel("Genes", fontsize=15)
        ax[i, j].set_ylabel("Frequency", fontsize=15)
        ax[i, j].yaxis.set_tick_params(labelsize="medium")
        ax[i, j].xaxis.set_tick_params(labelsize="medium")
save_img(fig, Im_dir_path, "Top8Genes9Classes.png")
