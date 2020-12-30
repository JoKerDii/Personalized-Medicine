"""
Data preprocessing for building machine learning models.
"""

### import libraries
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings("ignore")

### import data
for dirname, _, filenames in os.walk("../data/dataset"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv("../data/dataset/training_variants.zip")
print("Number of data points : ", data.shape[0])
print("Number of features : ", data.shape[1])
print("Features : ", data.columns.values)

data_text = pd.read_csv(
    "../data/dataset/training_text.zip",
    sep="\|\|",
    engine="python",
    names=["ID", "TEXT"],
    skiprows=1,
)
print("Number of data points : ", data_text.shape[0])
print("Number of features : ", data_text.shape[1])
print("Features : ", data_text.columns.values)

### data preprocessing
tokenizer = RegexpTokenizer("\w+'?\w+|\w+")
stop_words = stopwords.words("english")
exceptionStopWords = {
    "again",
    "against",
    "ain",
    "almost",
    "among",
    "amongst",
    "amount",
    "anyhow",
    "anyway",
    "aren",
    "aren't",
    "below",
    "bottom",
    "but",
    "cannot",
    "couldn",
    "couldn't",
    "didn",
    "didn't",
    "doesn",
    "doesn't",
    "don",
    "don't",
    "done",
    "down",
    "except",
    "few",
    "hadn",
    "hadn't",
    "hasn",
    "hasn't",
    "haven",
    "haven't",
    "however",
    "isn",
    "isn't",
    "least",
    "mightn",
    "mightn't",
    "move",
    "much",
    "must",
    "mustn",
    "mustn't",
    "needn",
    "needn't",
    "neither",
    "never",
    "nevertheless",
    "no",
    "nobody",
    "none",
    "noone",
    "nor",
    "not",
    "nothing",
    "should",
    "should've",
    "shouldn",
    "shouldn't",
    "too",
    "top",
    "up",
    "very" "wasn",
    "wasn't",
    "well",
    "weren",
    "weren't",
    "won",
    "won't",
    "wouldn",
    "wouldn't",
}
stop_words = set(stop_words).union(STOP_WORDS)
final_stop_words = stop_words - exceptionStopWords
nlp = spacy.load("en", disable=["parser", "tagger", "ner"])


def make_token(x):
    """ Tokenize the text (remove punctuations and spaces)"""
    return tokenizer.tokenize(str(x))


def remove_stopwords(x):
    return [token for token in x if token not in final_stop_words]


def lemmatization(x):
    lemma_result = []
    for words in x:
        doc = nlp(words)
        for token in doc:
            lemma_result.append(token.lemma_)
    return lemma_result


def pipeline(total_text, index, column):
    """ A pipeline to process text data """
    if type(total_text) is str:
        total_text = total_text.lower()
        total_text = make_token(total_text)
        total_text = remove_stopwords(total_text)
        total_text = lemmatization(total_text)
        string = " ".join(total_text)
        data_text[column][index] = string


for index, row in data_text.iterrows():
    if type(row["TEXT"]) is str:
        pipeline(row["TEXT"], index, "TEXT")
    else:
        print("there is no text description for id:", index)

### merge genes, variations and text data by ID
result = pd.merge(data, data_text, on="ID", how="left")
result.loc[result["TEXT"].isnull(), "TEXT"] = result["Gene"] + " " + result["Variation"]
result.Gene = result.Gene.str.replace("\s+", "_")
result.Variation = result.Variation.str.replace("\s+", "_")

## write to pickle
pd.to_pickle(result, "/home/zhendi/pm/scripts/result_non_split.pkl")
