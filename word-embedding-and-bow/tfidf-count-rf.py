# TFIDF + truncated or COUNT + truncated or TFIDF + COUNT + truncated --> RF
from __future__ import division, print_function

from builtins import range

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split


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
    print("MCC: {}".format(f1_score(y, preds)))


### Import Data
result = pd.read_pickle("/home/zhendi/pm/scripts/result_non_split_rmnum.pkl")
labels = result[["Class"]] - 1

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)

cvec = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)

# Try n_components between 360-390
svdT = TruncatedSVD(n_components=390, n_iter=5)


# tfidf.fit(result["TEXT"])
# cvec.fit(result['TEXT'])


def Vector_transformer(df, solver="TFIDF", truncate=True):
    """This is a function to extract features:
    - one-hot encoded genes and variations, and tfidf or count vectorization of text data.
    df argument should be a pandas dataframe with only Gene, Variation, TEXT, and Class columns"""
    # make a copy
    temp = df.copy()
    labels = temp["Class"] - 1
    del temp["Class"]

    # onehot encode gene and variation
    print("Onehot Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    # split the data to training data and testing data
    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        temp, labels, test_size=0.2, random_state=5, stratify=labels
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    svdT = TruncatedSVD(n_components=390, n_iter=5)

    if solver == "TFIDF":
        # Tfidf vectorize TEXT
        tfidf = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            analyzer="word",
            stop_words="english",
            token_pattern=r"\w+",
        )
        print("Tfidf Vectorizing...")
        svdT_tfidf_xtrain = svdT.fit_transform(tfidf.fit_transform(X_train["TEXT"]))

        svdT_tfidf_xtest = svdT.transform(tfidf.transform(X_test["TEXT"]))

    elif solver == "COUNT":
        # Count vectorize TEXT
        cvec = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            analyzer="word",
            stop_words="english",
            token_pattern=r"\w+",
        )
        print("Count Vectorizing...")
        svdT_tfidf_xtrain = svdT.fit_transform(cvec.fit_transform(X_train["TEXT"]))

        svdT_tfidf_xtest = svdT.transform(cvec.transform(X_test["TEXT"]))

    else:
        print("Solver not available.")

    del X_train["TEXT"]
    del X_test["TEXT"]

    # rename the colnames
    print("Renaming...")
    tempc_xtrain = list(X_train.columns)
    tempc_xtest = list(X_test.columns)

    for i in range(np.shape(svdT_tfidf_xtrain)[1]):
        tempc_xtrain.append("vec_" + str(i + 1))
    for i in range(np.shape(svdT_tfidf_xtest)[1]):
        tempc_xtest.append("vec_" + str(i + 1))

    X_train = pd.concat(
        [X_train, pd.DataFrame(svdT_tfidf_xtrain, index=X_train.index)], axis=1
    )
    X_test = pd.concat(
        [X_test, pd.DataFrame(svdT_tfidf_xtest, index=X_test.index)], axis=1
    )
    X_train.columns = tempc_xtrain
    X_test.columns = tempc_xtest

    print("Finished.")

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = Vector_transformer(
    result[["Gene", "Variation", "TEXT", "Class"]], solver="COUNT"
)
X_train, y_train, X_test, y_test = Vector_transformer(
    result[["Gene", "Variation", "TEXT", "Class"]], solver="TFIDF"
)
# pd.to_pickle((X_train, y_train, X_test, y_test), "/home/zhendi/pm/scripts/Onehot_tfidf_split.pkl")
# pd.to_pickle((X_train, y_train, X_test, y_test), "/home/zhendi/pm/scripts/Onehot_count_split.pkl")

evaluate_features(X_train, y_train)


# better way
def buildFeatures_split(df):
    """This is a function to extract all features:
    one-hot encoded genes and variations, tfidf, and count vectorization of text data.
    df argument should be a pandas dataframe with only Gene, Variation, TEXT, and Class columns"""

    # make a copy
    temp = df.copy()
    labels = temp["Class"] - 1
    del temp["Class"]

    # onehot encode gene and variation
    print("Onehot Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    # split the data to training data and testing data
    print("Splitting...")
    X_train, X_test, y_train, y_test = train_test_split(
        temp, labels, test_size=0.2, random_state=5, stratify=labels
    )
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    svdT = TruncatedSVD(n_components=390, n_iter=5)

    # Tfidf vectorize TEXT
    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        analyzer="word",
        stop_words="english",
        token_pattern=r"\w+",
    )
    print("Tfidf Vectorizing...")
    temp_tfidf_xtrain = svdT.fit_transform(tfidf.fit_transform(X_train["TEXT"]))
    print("xtrain tfidf shape: ", temp_tfidf_xtrain.shape)
    temp_tfidf_xtest = svdT.transform(tfidf.transform(X_test["TEXT"]))
    print("xtest tfidf shape: ", temp_tfidf_xtest.shape)

    # Count vectorize TEXT
    cvec = CountVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        analyzer="word",
        stop_words="english",
        token_pattern=r"\w+",
    )
    print("Count Vectorizing...")
    temp_count_xtrain = svdT.fit_transform(cvec.fit_transform(X_train["TEXT"]))
    print("xtrain cvec shape: ", temp_count_xtrain.shape)
    temp_count_xtest = svdT.transform(cvec.transform(X_test["TEXT"]))
    print("xtest cvec shape: ", temp_count_xtest.shape)

    del X_train["TEXT"]
    del X_test["TEXT"]

    # rename the colnames
    print("Renaming...")
    tempc_xtrain = list(X_train.columns)
    tempc_xtest = list(X_test.columns)

    for i in range(np.shape(temp_tfidf_xtrain)[1]):
        tempc_xtrain.append("tfidf_" + str(i + 1))
    for i in range(np.shape(temp_tfidf_xtest)[1]):
        tempc_xtest.append("tfidf_" + str(i + 1))

    for i in range(np.shape(temp_count_xtrain)[1]):
        tempc_xtrain.append("count_" + str(i + 1))
    for i in range(np.shape(temp_count_xtest)[1]):
        tempc_xtest.append("count_" + str(i + 1))

    X_train = pd.concat(
        [
            X_train,
            pd.DataFrame(temp_tfidf_xtrain, index=X_train.index),
            pd.DataFrame(temp_count_xtrain, index=X_train.index),
        ],
        axis=1,
    )
    X_test = pd.concat(
        [
            X_test,
            pd.DataFrame(temp_tfidf_xtest, index=X_test.index),
            pd.DataFrame(temp_count_xtest, index=X_test.index),
        ],
        axis=1,
    )
    X_train.columns = tempc_xtrain
    X_test.columns = tempc_xtest

    print("Finished.")

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = buildFeatures_split(
    result[["Gene", "Variation", "TEXT", "Class"]]
)
# pd.to_pickle(
#     (X_train, y_train, X_test, y_test),
#     "/home/zhendi/pm/scripts/Onehot_count_tfidf_split.pkl",
# )


## for text features only

# tfidf vectorized features
X_train, X_test, y_train, y_test = train_test_split(
    result, labels, test_size=0.2, random_state=5, stratify=labels
)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)

X_train = svdT.fit_transform(tfidf.fit_transform(X_train["TEXT"]))
y_train = svdT.transform(tfidf.transform(X_test["TEXT"]))

# count vectorized features
X_train, X_test, y_train, y_test = train_test_split(
    result, labels, test_size=0.2, random_state=5, stratify=labels
)

cvec = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)


X_train = svdT.fit_transform(cvec.fit_transform(X_train["TEXT"]))
y_train = svdT.transform(cvec.transform(X_test["TEXT"]))


## evaluate the features
evaluate_features(X_train, y_train)


## evaluation results

# not strict and all features
X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/Onehot_count_split.pkl"
)
evaluate_features(X_train, y_train)

X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/Onehot_tfidf_split.pkl"
)
evaluate_features(X_train, y_train)

X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/Onehot_count_tfidf_split.pkl"
)
evaluate_features(X_train, y_train)

# text features only
X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/tfidf_only_split.pkl"
)
evaluate_features(X_train, y_train)

X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/count_only_split.pkl"
)
evaluate_features(X_train, y_train)

X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/count_tfidf_only_split.pkl"
)

# for data, more stopword removed, and text features only
X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/count_only_split_strict.pkl"
)
evaluate_features(X_train, y_train)


X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/tfidf_only_split_strict.pkl"
)
evaluate_features(X_train, y_train)


X_train, y_train, X_test, y_test = pd.read_pickle(
    "/home/zhendi/pm/scripts/count_tfidf_only_split_strict.pkl"
)
evaluate_features(X_train, y_train)
