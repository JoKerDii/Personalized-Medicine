# EDA - Text Analysis

### Import Libraries

import os
import pickle
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

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


# Distribution of words and characters
n_words = result["TEXT"].apply(lambda x: len(str(x).split()))
fig = plt.figure(figsize=(12, 8))
sns.distplot(n_words.values, bins=100, kde=False, color="brown")
plt.xlabel("Number of Words", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title("Distribution of the Number of Words", fontsize=20)
save_img(fig, Im_dir_path, "DistributionOfWords.png")
plt.show()

n_chars = result["TEXT"].apply(lambda x: len(str(x)))
fig = plt.figure(figsize=(12, 8))
sns.distplot(n_chars.values, bins=100, kde=False, color="brown")
plt.xlabel("Number of Characters", fontsize=15)
plt.ylabel("Frequency", fontsize=15)
plt.title("Distribution of Number of Characters", fontsize=20)
save_img(fig, Im_dir_path, "DistributionOfChars.png")
plt.show()

### Text length distribution
result["n_words"] = result["TEXT"].apply(lambda x: len(str(x).split()))
fig = plt.figure(figsize=(12, 8))
sns.violinplot(x="Class", y="n_words", data=result, inner=None)
sns.swarmplot(x="Class", y="n_words", data=result, color="w", alpha=0.5)
plt.ylabel("Text Count", fontsize=15)
plt.xlabel("Class", fontsize=15)
plt.title("Text length distribution", fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
save_img(fig, Im_dir_path, "TextLength9Classes.png")
plt.show()

### Wordcloud for all the text
def plotWordCloud_forall():
    combText = result["TEXT"].agg(lambda x: " ".join(x.dropna()))
    wordcloud = WordCloud(
        background_color="white", colormap="Dark2", max_font_size=150, random_state=42
    ).generate(combText)
    # Display the generated image:
    print("word cloud for text ")
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("All Text Data")
    plt.axis("off")
    save_img(fig, Im_dir_path, "wordCloud_forall.png")
    plt.show()


plotWordCloud_forall()

### Wordcloud for each class
wc = WordCloud(
    background_color="white", colormap="Dark2", max_font_size=150, random_state=42
)


def plotWordCloud_foreach(class_n, model, TEXTdata):
    combText = TEXTdata.agg(lambda x: " ".join(x.dropna()))
    wordcloud = model.generate(combText)
    # Display the generated image:
    print("word cloud for Class: ", class_n)
    fig = plt.figure(figsize=(12, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title("Class: " + str(class_n), fontsize=15)
    plt.axis("off")
    filename = "wordCloud_class_" + str(class_n) + ".png"
    save_img(fig, Im_dir_path, filename)
    plt.show()


for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    plotWordCloud_foreach(class_n, wc, textdata)


### Distribution of Unigram, Bigram, and Trigram

# build the model
vec = CountVectorizer().fit(result["TEXT"])

# Non-strict for unigrams

result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
vec = CountVectorizer(ngram_range=(1, 1)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)
df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Unigram", "Frequency"])
    for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Unigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Unigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Unigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(fig, Im_dir_path, "TextFeature_Count_uni_Class" + str(i + 1) + ".png")
        plt.show()


plot_classfeats_h(df_list)

# Non-strict for bigrams
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
vec = CountVectorizer(ngram_range=(2, 2)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)

df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Bigram", "Frequency"]) for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Bigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Bigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Bigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(fig, Im_dir_path, "TextFeature_Count_bi_Class" + str(i + 1) + ".png")
        plt.show()


plot_classfeats_h(df_list)

# non-strict for trigrams
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
vec = CountVectorizer(ngram_range=(3, 3)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)

df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Trigram", "Frequency"])
    for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Trigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Trigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Trigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(fig, Im_dir_path, "TextFeature_Count_tri_Class" + str(i + 1) + ".png")
        plt.show()


plot_classfeats_h(df_list)

# Strict for unigrams
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_strict.pkl")
vec = CountVectorizer(ngram_range=(1, 1)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)

df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Unigram", "Frequency"])
    for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Unigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Unigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Unigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(
            fig, Im_dir_path, "TextFeature_Count_uni_strict_Class" + str(i + 1) + ".png"
        )
        plt.show()


plot_classfeats_h(df_list)


# Strict for bigrams
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_strict.pkl")
vec = CountVectorizer(ngram_range=(2, 2)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)

df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Bigram", "Frequency"]) for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Bigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Bigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Bigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(
            fig, Im_dir_path, "TextFeature_Count_bi_strict_Class" + str(i + 1) + ".png"
        )
        plt.show()


plot_classfeats_h(df_list)


# Strict for trigrams
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_strict.pkl")
vec = CountVectorizer(ngram_range=(3, 3)).fit(result["TEXT"])


def get_top_n_words(x, model, n):
    vec = model
    bow = vec.transform(x)
    sum_words = bow.sum(axis=0)
    vec.vocabulary_.items()  # give a dictionary
    words_freq = [
        (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
    ]  # a list
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)  # a sorted list
    return words_freq[:n]  # select numbers


dict_unigram = {}
for i in range(9):
    class_n = i + 1
    textdata = result[result["Class"] == class_n]["TEXT"]
    dict_unigram[class_n] = get_top_n_words(textdata, vec, 20)
df_list = [
    pd.DataFrame(dict_unigram[i + 1], columns=["Trigram", "Frequency"])
    for i in range(9)
]


def plot_classfeats_h(dfs):
    x = np.arange(len(dfs[0]))  # y axis ticks
    for i, df in enumerate(dfs):
        fig = plt.figure(figsize=(12, 10))
        plt.xlabel("Frequency", labelpad=16, fontsize=15)
        plt.ylabel("Trigram", labelpad=16, fontsize=15)
        plt.title("Top 20 Trigram in Class: " + str(i + 1), fontsize=20)
        plt.barh(df.Trigram, df.Frequency, align="center", color="#32acbf")
        plt.yticks(x, fontsize=12)
        plt.xticks(fontsize=12)
        plt.ylim([-1, x[-1] + 1])
        save_img(
            fig, Im_dir_path, "TextFeature_Count_tri_strict_Class" + str(i + 1) + ".png"
        )
        plt.show()


plot_classfeats_h(df_list)
