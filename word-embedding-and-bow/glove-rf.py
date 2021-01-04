# Glove(glove.840B.300d)

from __future__ import division, print_function

from builtins import range

import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
labels = result[["Class"]] - 1


class GloveVectorizer:
    def __init__(self):
        # load in pre-trained word vectors
        print("Loading word vectors...")
        word2vec = {}
        embedding = []
        idx2word = []
        with open("/data/zhendi/nlp/glove.840B.300d.txt") as f:
            # is just a space-separated text file in the format:
            # word vec[0] vec[1] vec[2] ...
            for line in f:
                values = line.split(" ")
                word = values[0]
                vec = np.asarray(values[1:], dtype="float32")
                word2vec[word] = vec
                embedding.append(vec)
                idx2word.append(word)
        print("Found %s word vectors." % len(word2vec))

        # save for later
        self.word2vec = word2vec
        self.embedding = np.array(embedding)
        self.word2idx = {v: k for k, v in enumerate(idx2word)}
        self.V, self.D = self.embedding.shape

    def fit(self, data):
        pass

    def transform(self, data):
        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.lower().split()  # lowercase words
            vecs = []
            for word in tokens:
                if word in self.word2vec:
                    vec = self.word2vec[word]
                    vecs.append(vec)
                else:
                    vec = np.random.uniform(-0.25, 0.25, self.D)
                    vecs.append(vec)
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print("Numer of samples with no words found: %s / %s" % (emptycount, len(data)))
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def build_onehot_Features(df):
    """This is a function to extract features, df argument should be
    a pandas dataframe with only Gene, Variation, and TEXT columns"""
    # make a copy
    temp = df.copy()

    # onehot encode gene and variation
    print("Onehot Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    # glove vectorize TEXT
    print("Glove Vectorizing...")
    glovevec = GloveVectorizer()
    temp_glove = glovevec.fit_transform(temp["TEXT"])
    del temp["TEXT"]

    # rename the colnames
    tempc = list(temp.columns)
    for i in range(np.shape(temp_glove)[1]):
        tempc.append("glove_" + str(i + 1))

    temp = pd.concat([temp, pd.DataFrame(temp_glove, index=temp.index)], axis=1)
    temp.columns = tempc

    return temp


trainDf = build_onehot_Features(result[["Gene", "Variation", "TEXT"]])
# pd.to_pickle(trainDf, "/home/zhendi/pm/scripts/Onehot_glove_trainDf.pkl")
# trainDf = pd.read_pickle("/home/zhendi/pm/scripts/Onehot_glove_trainDf.pkl")


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
