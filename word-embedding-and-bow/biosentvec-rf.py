# BioSentVec_RF

### Import Libraries
import os
import re
import warnings

import numpy as np
import pandas as pd
import sent2vec
import spacy
from keras.utils import np_utils
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from spacy.lang.en.stop_words import STOP_WORDS

warnings.filterwarnings("ignore")


### Import Data
for dirname, _, filenames in os.walk("/home/zhendi/pm/data/dataset"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = pd.read_csv(
    "/home/zhendi/pm/data/dataset/training_variants.zip", encoding="ISO-8859–1"
)
print("Number of data points : ", data.shape[0])
print("Number of features : ", data.shape[1])
print("Features : ", data.columns.values)

data_text = pd.read_csv(
    "/home/zhendi/pm/data/dataset/training_text.zip",
    sep="\|\|",
    engine="python",
    names=["ID", "TEXT"],
    skiprows=1,
    encoding="ISO-8859–1",
)
print("Number of data points : ", data_text.shape[0])
print("Number of features : ", data_text.shape[1])
print("Features : ", data_text.columns.values)


### Data Preprocessing
tokenizer = RegexpTokenizer(r"\w+'?\w+|\w+")
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


puncts = [
    ",",
    ".",
    '"',
    ":",
    ")",
    "(",
    "-",
    "!",
    "?",
    "|",
    ";",
    "'",
    "$",
    "&",
    "/",
    "[",
    "]",
    ">",
    "%",
    "=",
    "#",
    "*",
    "+",
    "\\",
    "•",
    "~",
    "@",
    "£",
    "·",
    "_",
    "{",
    "}",
    "©",
    "^",
    "®",
    "`",
    "<",
    "→",
    "°",
    "€",
    "™",
    "›",
    "♥",
    "←",
    "×",
    "§",
    "″",
    "′",
    "Â",
    "█",
    "½",
    "à",
    "…",
    "“",
    "★",
    "”",
    "–",
    "●",
    "â",
    "►",
    "−",
    "¢",
    "²",
    "¬",
    "░",
    "¶",
    "↑",
    "±",
    "¿",
    "▾",
    "═",
    "¦",
    "║",
    "―",
    "¥",
    "▓",
    "—",
    "‹",
    "─",
    "▒",
    "：",
    "¼",
    "⊕",
    "▼",
    "▪",
    "†",
    "■",
    "’",
    "▀",
    "¨",
    "▄",
    "♫",
    "☆",
    "é",
    "¯",
    "♦",
    "¤",
    "▲",
    "è",
    "¸",
    "¾",
    "Ã",
    "⋅",
    "‘",
    "∞",
    "∙",
    "）",
    "↓",
    "、",
    "│",
    "（",
    "»",
    "，",
    "♪",
    "╩",
    "╚",
    "³",
    "・",
    "╦",
    "╣",
    "╔",
    "╗",
    "▬",
    "❤",
    "ï",
    "Ø",
    "¹",
    "≤",
    "‡",
    "√",
]


def clean_text(x):
    x = str(x)
    for punct in puncts:
        if punct in x:
            x = x.replace(punct, "")
    return x


def pipeline(total_text, index, column):
    """A pipeline to process text data for BioSentVec"""

    if type(total_text) is str:
        all_text = []

        # sentence tokenizer and case lower
        for sent in sent_tokenize(total_text):
            text = " ".join(word_tokenize(sent))
            all_text.append(text.lower())

        all_sents = []
        for sent in all_text:
            # clean punctuations
            sent = clean_text(sent)
            # print(type(sent))
            sent = word_tokenize(sent)
            sent = remove_stopwords(sent)
            sent = lemmatization(sent)
            string = " ".join(sent)
            # clean numbers
            sent_nonum = re.sub(r"\b\d+\b", "", string)
            string = " ".join(sent_nonum.split())

            all_sents.append(string)

        data_text["TEXT"][index] = all_sents


for index, row in data_text.iterrows():
    if type(row["TEXT"]) is str:
        pipeline(row["TEXT"], index, "TEXT")
    else:
        print("there is no text description for id:", index)


### Merge genes, variations and text data by ID
result = pd.merge(data, data_text, on="ID", how="left")
result.loc[result["TEXT"].isnull(), "TEXT"] = result["Gene"] + " " + result["Variation"]
result.Gene = result.Gene.str.replace("\s+", "_")
result.Variation = result.Variation.str.replace("\s+", "_")
labels = result[["Class"]] - 1

# save result
# pd.to_pickle(result, "/home/zhendi/pm/scripts/result_sentVec_strict.pkl")
# load result
# result = pd.read_pickle("/home/zhendi/pm/scripts/result_sentVec_strict.pkl")

# Define the model
import sent2vec

model_path = "/data/zhendi/nlp/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
model = sent2vec.Sent2vecModel()

try:
    model.load_model(model_path)
except Exception as e:
    print(e)
print("model successfully loaded")


def BioSentVec_transform(model, data):

    # determine the dimensionality of vectors
    V = model.embed_sentence("once upon a time .")
    D = V.shape[1]
    print("D = V.shape[1]: ", D)
    X = np.zeros((len(data), D))

    emptycount = 0
    n = 0
    for record in data:
        try:
            vec = model.embed_sentences(record)
        except KeyError:
            print("there is a sent with no match.")
            pass
        if len(vec) > 0:
            X[n] = vec.mean(axis=0)
        else:
            emptycount += 1
        n += 1

    print("Number of samples with no words found: %s / %s" % (emptycount, len(data)))
    return X


def build_onehot_Features(df):
    """This is a function to extract features, df argument should be
    a pandas dataframe with only Gene, Variation, and TEXT columns"""
    # make a copy
    temp = df.copy()

    # onehot encode gene and variation
    print("Onehot Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    # Sent2Vec vectorize TEXT
    print("Sent2Vec Vectorizing...")

    temp_sent2v = BioSentVec_transform(temp["TEXT"])
    del temp["TEXT"]

    # rename the colnames
    tempc = list(temp.columns)
    for i in range(np.shape(temp_sent2v)[1]):
        tempc.append("sent2v_" + str(i + 1))

    temp = pd.concat([temp, pd.DataFrame(temp_sent2v, index=temp.index)], axis=1)
    temp.columns = tempc

    return temp


trainDf = build_onehot_Features(result[["Gene", "Variation", "TEXT"]])

# save
# pd.to_pickle(trainDf, "/home/zhendi/pm/scripts/Onehot_biosentvec_trainDf_strict.pkl")

# laod
# trainDf = pd.read_pickle("/home/zhendi/pm/scripts/Onehot_biosentvec_trainDf_strict.pkl")


### Split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    trainDf, labels, test_size=0.2, random_state=5, stratify=labels
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
encoded_test_y = np_utils.to_categorical((le.inverse_transform(y_test)))


def evaluate_features(X, y, clf=None):

    if clf is None:
        clf = RandomForestClassifier(n_estimators=400, random_state=5)

    probas = cross_val_predict(
        clf,
        X,
        y,
        cv=StratifiedKFold(n_splits=3),
        n_jobs=-1,
        method="predict_proba",
        verbose=2,
    )
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]
    print("Log loss: {}".format(log_loss(y, probas)))
    print("Accuracy: {}".format(balanced_accuracy_score(y, preds)))
    print("F1 score: {}".format(f1_score(y, preds, average="micro")))


evaluate_features(X_train, y_train)
