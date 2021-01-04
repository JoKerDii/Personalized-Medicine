# BioWordVec Embedding + RF


from __future__ import division, print_function

from builtins import range

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, log_loss, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder

result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
labels = result[["Class"]] - 1


class BioWordVecVectorizer:
    def __init__(self):
        print("Loading in word vectors...")
        self.word_vectors = KeyedVectors.load_word2vec_format(
            "/data/zhendi/nlp/BioWordVec_PubMed_MIMICIII_d200.vec.bin", binary=True
        )  # 200 dim
        print("Finished loading in word vectors")
        self.features = None

    def fit(self, data):
        pass

    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector("the")
        self.D = v.shape[0]
        print("The dimension is ", self.D)

        X = np.zeros((len(data), self.D))
        n = 0
        features = {}
        emptycount = 0
        for sentence in data:
            tokens = (
                sentence.split()
            )  # do not lowercase the words as word2vec has uppercase words
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)

                    if word not in features.keys():
                        features[word] = 1
                    else:
                        features[word] += 1
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        print(
            "Number of samples with no words found: %s / %s" % (emptycount, len(data))
        )
        return X

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def get_model_feature_names(self):
        return list(self.word_vectors.vocab.keys())


def build_onehot_Features(df):
    """This is a function to extract features, df argument should be
    a pandas dataframe with only Gene, Variation, and TEXT columns"""
    # make a copy
    temp = df.copy()

    # onehot encode gene and variation
    print("Onehot Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    # Word2Vec vectorize TEXT
    print("Word2Vec Vectorizing...")
    # glovevec = GloveVectorizer()
    w2v = BioWordVecVectorizer()
    temp_w2v = w2v.fit_transform(temp["TEXT"])
    del temp["TEXT"]

    # rename the colnames
    tempc = list(temp.columns)
    for i in range(np.shape(temp_w2v)[1]):
        tempc.append("w2v_" + str(i + 1))

    temp = pd.concat([temp, pd.DataFrame(temp_w2v, index=temp.index)], axis=1)
    temp.columns = tempc

    return temp


trainDf = build_onehot_Features(result[["Gene", "Variation", "TEXT"]])
# pd.to_pickle(trainDf, "/home/zhendi/pm/scripts/Onehot_biowordvec_trainDf.pkl")
# trainDf = pd.read_pickle("/home/zhendi/pm/scripts/Onehot_biowordvec_trainDf.pkl")

### Split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    trainDf, labels, test_size=0.2, random_state=5
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
    print("MCC: {}".format(matthews_corrcoef(y, preds)))


evaluate_features(X_train, y_train)
