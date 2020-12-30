### feature extraction
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

result = pd.read_pickle("result_non_split.pkl")

maxFeats = 10000

tfidf = TfidfVectorizer(
    min_df=5,
    max_features=maxFeats,
    ngram_range=(1, 2),
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)
tfidf.fit(result["TEXT"])

cvec = CountVectorizer(
    min_df=5,
    ngram_range=(1, 2),
    max_features=maxFeats,
    analyzer="word",
    stop_words="english",
    token_pattern=r"\w+",
)
cvec.fit(result["TEXT"])

# try n_components between 360-390
svdT = TruncatedSVD(n_components=390, n_iter=5)
svdTFit = svdT.fit_transform(tfidf.transform(result["TEXT"]))


def buildFeatures(df):
    """This is a function to extract features, df argument should be
    a pandas dataframe with only Gene, Variation, and TEXT columns"""

    temp = df.copy()

    print("Encoding...")
    temp = pd.get_dummies(temp, columns=["Gene", "Variation"], drop_first=True)

    print("TFIDF...")
    temp_tfidf = tfidf.transform(temp["TEXT"])

    print("Count Vecs...")
    temp_cvec = cvec.transform(temp["TEXT"])

    print("Latent Semantic Analysis Cols...")
    del temp["TEXT"]

    tempc = list(temp.columns)

    temp_lsa_tfidf = svdT.transform(temp_tfidf)
    temp_lsa_cvec = svdT.transform(temp_cvec)

    for i in range(np.shape(temp_lsa_tfidf)[1]):
        tempc.append("lsa_t" + str(i + 1))
    for i in range(np.shape(temp_lsa_cvec)[1]):
        tempc.append("lsa_c" + str(i + 1))
    temp = pd.concat(
        [
            temp,
            pd.DataFrame(temp_lsa_tfidf, index=temp.index),
            pd.DataFrame(temp_lsa_cvec, index=temp.index),
        ],
        axis=1,
    )
    return temp, tempc


trainDf, traincol = buildFeatures(result[["Gene", "Variation", "TEXT"]])
trainDf.columns = traincol
# pd.to_pickle(trainDf, "trainDf.pkl")


# better way
def buildFeatures_split(df):
    """This is a function to extract features, df argument should be
    a pandas dataframe with only Gene, Variation, TEXT, and Class columns"""
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

    # Tfidf vectorize TEXT
    tfidf = TfidfVectorizer(
        max_features=maxFeats,
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
        max_features=maxFeats,
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
# pd.to_pickle((X_train, y_train, X_test, y_test), "/home/zhendi/pm/scripts/Onehot_tfidf_split.pkl")
# pd.to_pickle(
#     (X_train, y_train, X_test, y_test),
#     "/home/zhendi/pm/scripts/Onehot_count_tfidf_split.pkl",
# )
