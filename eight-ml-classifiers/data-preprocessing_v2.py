# DataPreprocessing_update
"""
Data preprocessing for building machine learning models.
"""

### import libraries
import os
import pickle
import re
import warnings

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings("ignore")
### Import Data
for dirname, _, filenames in os.walk("/home/zhendi/pm/data/dataset"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


data = pd.read_csv("/home/zhendi/pm/data/dataset/training_variants.zip")
print("Number of data points : ", data.shape[0])
print("Number of features : ", data.shape[1])
print("Features : ", data.columns.values)

data_text = pd.read_csv(
    "/home/zhendi/pm/data/dataset/training_text.zip",
    sep="\|\|",
    engine="python",
    names=["ID", "TEXT"],
    skiprows=1,
)
print("Number of data points : ", data_text.shape[0])
print("Number of features : ", data_text.shape[1])
print("Features : ", data_text.columns.values)

### data preprocessing
tokenizer = RegexpTokenizer(r"\w+'?\w+|\w+")
stop_words = stopwords.words("english")

stop_words = set(stop_words).union(STOP_WORDS)
stopwords_extra = {
    # "line",
    "fig",
    "figure",
    "tab",
    "table",
    "author",
    # "find",
    "supplementary",
    "supplement",
    "et",
    "al",
    # "evaluate",
    # "show",
    # "demonstrate",
    # "conclusion",
    # "study",
    # "analysis",
    # "method",
}
stopwords_for_unigram_bigram = {
    "cancer",
    "mutant",
    "mutation",
    "variant",
    "sequence",
    "sample",
    "protein",
    "expression",
    "patient",
    "gene",
    "cell",
    "assay",
    "wild",
    "type",
    "amino",
    "acid",
    "tumor",
}

stop_words.update(stopwords_for_unigram_bigram)
stop_words.update(stopwords_extra)
nlp = spacy.load("en", disable=["parser", "tagger", "ner"])


def make_token(x):
    """ Tokenize the text (remove punctuations and spaces)"""
    return tokenizer.tokenize(str(x))


def remove_stopwords(x):
    return [token for token in x if token not in stop_words]


def lemmatization(x):
    lemma_result = []
    for words in x:
        doc = nlp(words)
        for token in doc:
            lemma_result.append(token.lemma_)
    return lemma_result


def pipeline(total_text, index, column):
    """A pipeline to process text data"""

    # if type(total_text) is str:
    total_text1 = total_text.lower()
    total_text2 = make_token(total_text1)
    total_text3 = lemmatization(total_text2)  # lemmatization before removing stop words
    total_text4 = remove_stopwords(total_text3)
    string = " ".join(total_text4)  # correct

    # remove numbers
    string1 = re.sub(r"\b\d+\b", " ", string)
    string2 = re.sub(r"\b\d+\w\b", " ", string1)
    # remove single char
    string3 = re.sub(r"\b\d\b", " ", string2)
    string4 = re.sub(r"\b\w\b", " ", string3)

    string5 = make_token(string4)
    string6 = remove_stopwords(string5)
    new_text = " ".join(string6)
    data_text[column][index] = new_text


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
pd.to_pickle(result, "/home/zhendi/pm/scripts/result_non_split_strict.pkl")
