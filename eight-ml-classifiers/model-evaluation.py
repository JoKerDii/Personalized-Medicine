# import libraries
import os
import pickle
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# import pre-processed data
result = pd.read_pickle("pm/text-ml-classification/scripts/result_non_split.pkl")
trainDf = pd.read_pickle("/home/zhen.di/pm/text-ml-classification/scripts/trainDf.pkl")

# split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    trainDf, result.Class, test_size=0.2, random_state=5, stratify=result.Class
)

# encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)  # int64
y_test = le.transform(y_test)
encoded_test_y = np_utils.to_categorical((le.inverse_transform(y_test)))


def plot_confusion_matrix(clf, test_y, predict_y):
    """ Give confusion matrix based on testing data """

    C = confusion_matrix(test_y, predict_y)

    labels = le.classes_
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        C, annot=True, cmap="Blues", fmt=".0f", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Class", fontsize=20)
    plt.ylabel("Original Class", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    Im_dir_path = "/home/zhen.di/pm/text-ml-classification/scripts/image/cm/"
    clean_name = re.split("[()]", str(clf))[0]
    Name_Formatted = ("%s" % clean_name) + "_cm.png"
    file_path = os.path.join(Im_dir_path, Name_Formatted)
    fig.savefig(file_path)

    plt.show()


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Generate a simple plot of the test and training learning curves"""

    fig = plt.figure(figsize=(8, 8))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples", fontsize=20)
    plt.ylabel("Score", fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )

    plt.legend(loc="best", prop={"size": 15})

    Im_dir_path = "/home/zhen.di/pm/text-ml-classification/scripts/image/"
    clean_name = re.split("[()]", str(estimator))[0]
    Name_Formatted = ("%s" % clean_name) + "_lc.png"
    file_path = os.path.join(Im_dir_path, Name_Formatted)
    fig.savefig(file_path)

    plt.show()
    return plt


def mea_metrics_calc(num, model, train, test, target, target_test):
    """ For the calculation and storage of accuracy and log loss """

    global mea_all

    ytrain = model.predict(train)
    yprobas = model.predict_proba(train)
    ytest = model.predict(test)
    yprobas_test = model.predict_proba(test)
    print("target = ", target[:5])
    print("ytrain = ", ytrain[:5])
    print("target_test =", target_test[:5])
    print("ytest =", ytest[:5])

    num_mea = 0
    for x in metrics_now:
        if x == 1:
            # log loss
            mea_train = log_loss(target, yprobas)
            mea_test = log_loss(target_test, yprobas_test)
        elif x == 2:
            # accuracy
            mea_train = round(balanced_accuracy_score(target, ytrain) * 100, 3)
            mea_test = round(balanced_accuracy_score(target_test, ytest) * 100, 3)
        elif x == 3:
            # f1 score
            mea_train = f1_score(target, ytrain, average="micro")
            mea_test = f1_score(target_test, ytest, average="micro")

        print("Measure of", metrics_all[x], "for train =", mea_train)
        print("Measure of", metrics_all[x], "for test =", mea_test)

        mea_all[num_mea].append(mea_train)  # train
        mea_all[num_mea + 1].append(mea_test)  # test
        num_mea += 2

    return plot_confusion_matrix(model, target_test, ytest)


def evaluate_features(
    X, y, X_test, y_test, clf=None, kfold=StratifiedKFold(n_splits=3)
):
    """Can be used to evaluate features on training data and testing data;
    also compare model performance when specifying clf"""
    if clf is None:
        clf = RandomForestClassifier(n_estimators=400, random_state=5, n_jobs=-1)

    probas = cross_val_predict(
        clf,
        X,
        y,
        cv=kfold,
        n_jobs=-1,
        method="predict_proba",
        verbose=2,
    )
    pred_indices = np.argmax(probas, axis=1)
    classes = np.unique(y)
    preds = classes[pred_indices]

    print("Cross validation on training data: ")
    print("Log loss: {}".format(log_loss(y, probas)))
    print("Accuracy: {}".format(balanced_accuracy_score(y, preds)))
    print("MCC: {}".format(f1_score(y, preds, average="micro")))

    print("Validation on testing data: ")
    clf.fit(X, y)
    ytest = clf.predict(X_test)
    yprobas_test = clf.predict_proba(X_test)
    print("Log loss: {}".format(log_loss(y_test, yprobas_test)))
    print("Accuracy: {}".format(balanced_accuracy_score(y_test, ytest)))
    print("MCC: {}".format(f1_score(y_test, ytest, average="micro")))


def all_metrics(num, model, train, test, target, target_test):
    """ Calculating metric scores for all models"""

    ytrain = model.predict(train)
    yprobas = model.predict_proba(train)

    ytest = model.predict(test)
    yprobas_test = model.predict_proba(test)

    logloss_train = log_loss(target, yprobas)
    logloss_test = log_loss(target_test, yprobas_test)

    print("Training Log Loss: ", logloss_train)
    print("Testing Log Loss: ", logloss_test)

    acc_train = round(balanced_accuracy_score(target, ytrain) * 100, 3)
    acc_test = round(balanced_accuracy_score(target_test, ytest) * 100, 3)

    print("Training Accuracy: ", acc_train)
    print("Testing Accuracy: ", acc_test)

    f1score_train = f1_score(target, ytrain, average="micro")
    f1score_test = f1_score(target_test, ytest, average="micro")

    print("Training f1 Score: ", f1score_train)
    print("Testing f1 Score: ", f1score_test)


def evaluate_resampling(X, y, X_test, y_test, clf=None):
    """For evaluating various resampling methods"""
    if clf is None:
        clf = RandomForestClassifier(n_estimators=400, random_state=5, n_jobs=-1)

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

    print("Cross validation on training data: ")
    print("Log loss: {}".format(log_loss(y, probas)))
    print("Accuracy: {}".format(balanced_accuracy_score(y, preds)))
    print("F1 score: {}".format(f1_score(y, preds, average="micro")))

    print("Validation on testing data: ")
    clf.fit(X, y)
    ytest = clf.predict(X_test)
    yprobas_test = clf.predict_proba(X_test)
    print("Log loss: {}".format(log_loss(y_test, yprobas_test)))
    print("Accuracy: {}".format(balanced_accuracy_score(y_test, ytest)))
    print("F1 score: {}".format(f1_score(y_test, ytest, average="micro")))


# import trained models
filename = "/home/zhendi/pm/scripts/ML_model/pipe/svr_CV_best.sav"
with (open(filename, "rb")) as openfile:
    svr_CV_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/logreg_CV_best.sav"
with (open(filename, "rb")) as openfile:
    logreg_CV_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/knn_CV_best.sav"
with (open(filename, "rb")) as openfile:
    knn_CV_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/random_forest_CV_best.sav"
with (open(filename, "rb")) as openfile:
    random_forest_CV_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/Ada_Boost_CV_best.sav"
with (open(filename, "rb")) as openfile:
    Ada_Boost_CV_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/xgb_clf_cv_best.sav"
with (open(filename, "rb")) as openfile:
    xgb_clf_cv_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/mlp_GS_best.sav"
with (open(filename, "rb")) as openfile:
    mlp_GS_best = pickle.load(openfile)

filename = "/home/zhendi/pm/scripts/ML_model/pipe/Voting_ens.sav"
with (open(filename, "rb")) as openfile:
    Voting_ens = pickle.load(openfile)


# pre settings

kfold = StratifiedKFold(n_splits=5)
random_state = 0
metrics_all = {1: "Log_Loss", 2: "Accuracy", 3: "F1_score"}
metrics_now = [1, 2, 3]
num_models = 8
mea_train = []
mea_test = []
mea_all = np.empty((len(metrics_now) * 2, 0)).tolist()

# best parameters
print(svr_CV_best)
print(logreg_CV_best)
print(knn_CV_best)
print(random_forest_CV_best)
print(Ada_Boost_CV_best)
print(xgb_clf_cv_best)
print(mlp_GS_best)
print(Voting_ens)

# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(0, svr_CV_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(1, logreg_CV_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(2, knn_CV_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(3, random_forest_CV_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(4, Ada_Boost_CV_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(5, xgb_clf_cv_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(6, mlp_GS_best, X_train, X_test, y_train, y_test)
mea_metrics_calc(7, Voting_ens, X_train, X_test, y_train, y_test)

# plot learning curve of model (with title = None)
plot_learning_curve(svr_CV_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(logreg_CV_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(knn_CV_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(random_forest_CV_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(Ada_Boost_CV_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(xgb_clf_cv_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(mlp_GS_best, None, X_train, y_train, cv=kfold)
plot_learning_curve(Voting_ens, None, X_train, y_train, cv=kfold)

# specify model names
models = pd.DataFrame(
    {
        "Model": [
            "SVM Classifier",
            "Logistic Regression Classifier",
            "kNN Classifier",
            "Random Forest Classifier",
            "AdaBoost Classifier",
            "XGBoost Classifier",
            "MLP Classifier",
            "Voting Classifier",
        ]
    }
)


# give accuracy and log loss for all models
for x in metrics_now:
    xs = metrics_all[x]
    models[xs + "_train"] = mea_all[(x - 1) * 2]
    models[xs + "_test"] = mea_all[(x - 1) * 2 + 1]
    if xs == "Accuracy":
        models[xs + "_diff"] = models[xs + "_train"] - models[xs + "_test"]
# save the model
filename = "ML_model/models8.pkl"
models.to_pickle(filename)

# generate performance table in order
# order the model performance by f1 score
print("In the descending order of F1 scores on testing data")
ms = metrics_all[metrics_now[2]]  # the F1 score
models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=False)

# save the model
filename = "ML_model/models8_s_f1.pkl"
models.to_pickle(filename)

# order the model performance by accuracy
print("In the descending order of balanced accuracy on testing data")
ms = metrics_all[metrics_now[1]]  # the accuracy
models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=False)

# save the model
filename = "ML_model/models8_s_acc.pkl"
models.to_pickle(filename)

# order the model performance by log loss
print("In the ascending order of log loss on testing data")
ms = metrics_all[metrics_now[0]]  # the log loss
models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=True)

# save the model
filename = "ML_model/models8_s_logloss.pkl"
models.to_pickle(filename)

# plot accuracy and log loss for all models
pd.options.display.float_format = "{:,.3f}".format
for x in metrics_now:
    # Plot
    xs = metrics_all[x]
    xs_train = metrics_all[x] + "_train"
    xs_test = metrics_all[x] + "_test"
    fig = plt.figure(figsize=[12, 6])

    xx = models["Model"]
    plt.tick_params(labelsize=14)
    plt.plot(xx, models[xs_train], marker="o", label=xs_train)
    plt.plot(xx, models[xs_test], marker="o", label=xs_test)
    plt.legend(prop={"size": 15})
    plt.title(
        str(xs) + " of " + str(num_models) + " popular models on train and test data",
        fontsize=15,
    )
    plt.xlabel("Models", fontsize=20)
    plt.ylabel(xs + ", %", fontsize=20)
    plt.xticks(xx, rotation=25, fontsize=15)
    # store the plots
    Im_dir_path = "/home/zhendi/pm/scripts/image/"
    # clean_name = re.split('[()]', str(estimator))[0]
    Name_Formatted = ("%s" % metrics_all[x]) + "_8models_ba.png"
    file_path = os.path.join(Im_dir_path, Name_Formatted)
    fig.savefig(file_path)

    plt.show()
