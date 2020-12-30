### import Libraries
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from keras.utils import np_utils
from sklearn.ensemble import (
    AdaBoostClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.metrics.classification import log_loss
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

result = pd.read_pickle("result_non_split.pkl")
labels = result.Class - 1
trainDf = pd.read_pickle("trainDf.pkl")


### split data into training data and testing data
X_train, X_test, y_train, y_test = train_test_split(
    trainDf, labels, test_size=0.2, random_state=5, stratify=labels
)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
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
            # balanced accuracy
            mea_train = round(balanced_accuracy_score(target, ytrain) * 100, 3)
            mea_test = round(balanced_accuracy_score(target_test, ytest) * 100, 3)
        elif x == 3:
            # f1 score
            mea_train = f1_score(target, ytrain, average="micro")
            mea_test = f1_score(target, ytest, average="micro")

        print("Measure of", metrics_all[x], "for train =", mea_train)
        print("Measure of", metrics_all[x], "for test =", mea_test)
        mea_all[num_mea].append(mea_train)  # train
        mea_all[num_mea + 1].append(mea_test)  # test
        num_acc += 2

    return plot_confusion_matrix(model, target_test, ytest)


# pre settings
kfold = StratifiedKFold(n_splits=5)
random_state = 0
metrics_all = {1: "Log_Loss", 2: "Accuracy", 3: "F1_score"}
metrics_now = [1, 2, 3]
num_models = 8
mea_train = []
mea_test = []
mea_all = np.empty((len(metrics_now) * 2, 0)).tolist()


### 1. Support Vector Machines
svr = SVC(probability=True)
# Hyperparameter tuning - Grid search cross validation
svr_CV = GridSearchCV(
    svr,
    param_grid={
        "C": [0.1, 1, 10, 100],
        "gamma": [1, 0.1, 0.01, 0.001],
        "kernel": ["poly", "rbf", "sigmoid", "linear"],
        "tol": [1e-4],
    },
    cv=kfold,
    verbose=False,
    n_jobs=-1,
)
svr_CV.fit(X_train, y_train)
svr_CV_best = svr_CV.best_estimator_
print("Best score: %0.3f" % svr_CV.best_score_)
print("Best parameters set:", svr_CV.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(0, svr_CV_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(svr_CV_best, "Support Vector Machines", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/svr_CV_best.sav"
pickle.dump(svr_CV_best, open(filename, "wb"))
## load the model
# with (open(filename, "rb")) as openfile:
# svr_CV_best = pickle.load(openfile)


### 2. Logistic Regression
logreg = LogisticRegression(multi_class="multinomial")
# Hyperparameter tuning - Grid search cross validation
logreg_CV = GridSearchCV(
    estimator=logreg,
    param_grid={"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]},
    cv=kfold,
    verbose=False,
)
logreg_CV.fit(X_train, y_train)
logreg_CV_best = logreg_CV.best_estimator_
print("Best score: %0.3f" % logreg_CV.best_score_)
print("Best parameters set:", logreg_CV.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(1, logreg_CV_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(logreg_CV_best, "Logistic Regression", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/logreg_CV_best.sav"
pickle.dump(logreg_CV_best, open(filename, "wb"))


### 3. k-Nearest Neighbors
knn = KNeighborsClassifier()
# Hyperparameter tuning - Grid search cross validation
param_grid = {"n_neighbors": range(2, 10)}
knn_CV = GridSearchCV(
    estimator=knn, param_grid=param_grid, cv=kfold, verbose=False
).fit(X_train, y_train)
knn_CV_best = knn_CV.best_estimator_
print("Best score: %0.3f" % knn_CV.best_score_)
print("Best parameters set:", knn_CV.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(2, knn_CV_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(knn_CV_best, "KNN", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/knn_CV_best.sav"
pickle.dump(knn_CV_best, open(filename, "wb"))

### 4. Random Forest
random_forest = RandomForestClassifier()
param_grid = {
    "bootstrap": [True, False],
    "max_depth": [5, 8, 10, 20, 40, 50, 60, 80, 100],
    "max_features": ["auto", "sqrt"],
    "min_samples_leaf": [1, 2, 4, 10, 20, 30, 40],
    "min_samples_split": [2, 5, 10],
    "n_estimators": [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
}
random_forest_CV = GridSearchCV(
    estimator=random_forest, param_grid=param_grid, cv=kfold, verbose=False, n_jobs=-1
)
random_forest_CV.fit(X_train, y_train)
random_forest_CV_best = random_forest_CV.best_estimator_
print("Best score: %0.3f" % random_forest_CV.best_score_)
print("Best parameters set:", random_forest_CV.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(3, random_forest_CV_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(
    random_forest_CV_best, "Random Forest Classifier", X_train, y_train, cv=kfold
)
# save the model
filename = "ML_model/pipe/random_forest_CV_best.sav"
pickle.dump(random_forest_CV_best, open(filename, "wb"))

### 5. Adaboost
param_grid = {
    "base_estimator__max_depth": [5, 10, 20, 50, 100, 150, 200],
    "n_estimators": [100, 500, 1000, 1500, 2000],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1.0],
    "algorithm": ["SAMME", "SAMME.R"],
}
Ada_Boost = AdaBoostClassifier(DecisionTreeClassifier())
Ada_Boost_CV = GridSearchCV(
    estimator=Ada_Boost, param_grid=param_grid, cv=kfold, verbose=False, n_jobs=10
)
Ada_Boost_CV.fit(X_train, y_train)
Ada_Boost_CV_best = Ada_Boost_CV.best_estimator_
print("Best score: %0.3f" % Ada_Boost_CV.best_score_)
print("Best parameters set:", Ada_Boost_CV.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(4, Ada_Boost_CV_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(
    Ada_Boost_CV_best, "AdaBoost Classifier", X_train, y_train, cv=kfold
)
# save the model
filename = "ML_model/pipe/Ada_Boost_CV_best.sav"
pickle.dump(Ada_Boost_CV_best, open(filename, "wb"))

### 6. XGBoost
xgb_clf = xgb.XGBClassifier(objective="multi:softprob")
parameters = {
    "n_estimators": [200, 300, 400],
    "learning_rate": [0.001, 0.003, 0.005, 0.006, 0.01],
    "max_depth": [4, 5, 6],
}
xgb_clf_cv = GridSearchCV(
    estimator=xgb_clf, param_grid=parameters, n_jobs=-1, cv=kfold
).fit(X_train, y_train)
xgb_clf_cv_best = xgb_clf_cv.best_estimator_
print("Best score: %0.3f" % xgb_clf_cv.best_score_)
print("Best parameters set:", xgb_clf_cv.best_params_)
mea_metrics_calc(5, xgb_clf_cv_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(xgb_clf_cv_best, "XGBoost Classifier", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/xgb_clf_cv_best.sav"
pickle.dump(xgb_clf_cv_best, open(filename, "wb"))


### 7. MLPClassifier
mlp = MLPClassifier()
param_grid = {
    "hidden_layer_sizes": [i for i in range(5, 25, 5)],
    "solver": ["sgd", "adam", "lbfgs"],
    "learning_rate": ["constant", "adaptive"],
    "max_iter": [500, 1000, 1200, 1400, 1600, 1800, 2000],
    "alpha": [10.0 ** (-i) for i in range(-3, 6)],
    "activation": ["tanh", "relu"],
}
mlp_GS = GridSearchCV(mlp, param_grid=param_grid, n_jobs=-1, cv=kfold, verbose=False)
mlp_GS.fit(X_train, y_train)
mlp_GS_best = mlp_GS.best_estimator_
print("Best score: %0.3f" % mlp_GS.best_score_)
print("Best parameters set:", mlp_GS.best_params_)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(6, mlp_GS_best, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(mlp_GS_best, "MLP Classifier", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/mlp_GS_best.sav"
pickle.dump(mlp_GS_best, open(filename, "wb"))


### 8. Voting Classifier
Voting_ens = VotingClassifier(
    estimators=[
        ("log", logreg_CV_best),
        ("rf", random_forest_CV_best),
        ("knn", knn_CV_best),
        ("svm", svr_CV_best),
    ],
    n_jobs=-1,
    voting="soft",
)
Voting_ens.fit(X_train, y_train)
# calculate log loss and accuracy and plot confusion matrix
mea_metrics_calc(7, Voting_ens, X_train, X_test, y_train, y_test)
# plot learning curve of model
plot_learning_curve(Voting_ens, "EnsembleVoting_ens", X_train, y_train, cv=kfold)
# save the model
filename = "ML_model/pipe/Voting_ens.sav"
pickle.dump(Voting_ens, open(filename, "wb"))


# ### Collecting results
# models = pd.DataFrame(
#     {
#         "Model": [
#             "SVM Classifier",
#             "Logistic Regression Classifier",
#             "k Nearest Neighbor Classifier",
#             "Random Forest Classifier",
#             "AdaBoost Classifier",
#             "XGBoost Classifier",
#             "MLP Classifier",
#             "Voting Classifier",
#         ]
#     }
# )

# # Give accuracy and log loss for all models
# for x in metrics_now:
#     xs = metrics_all[x]
#     models[xs + "_train"] = mea_all[(x - 1) * 2]
#     models[xs + "_test"] = mea_all[(x - 1) * 2 + 1]
#     if xs == "Accuracy":
#         models[xs + "_diff"] = models[xs + "_train"] - models[xs + "_test"]
# # Save the model
# filename = "ML_model/models8.pkl"
# models.to_pickle(filename)

# # Generate performance table in order
# # Order the model performance by f1 score
# print("In the descending order of F1 scores on testing data")
# ms = metrics_all[metrics_now[2]]  # the F1 score
# models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=False)

# # Save the model
# filename = "ML_model/models8_s_f1.pkl"
# models.to_pickle(filename)

# # Order the model performance by accuracy
# print("In the descending order of balanced accuracy on testing data")
# ms = metrics_all[metrics_now[1]]  # the accuracy
# models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=False)

# # Save the model
# filename = "ML_model/models8_s_acc.pkl"
# models.to_pickle(filename)

# # Order the model performance by log loss
# print("In the ascending order of log loss on testing data")
# ms = metrics_all[metrics_now[0]]  # the log loss
# models.sort_values(by=[(ms + "_test"), (ms + "_train")], ascending=True)

# # Save the model
# filename = "ML_model/models8_s_logloss.pkl"
# models.to_pickle(filename)

# # Plot accuracy and log loss for all models
# pd.options.display.float_format = "{:,.3f}".format
# for x in metrics_now:
#     # Plot
#     xs = metrics_all[x]
#     xs_train = metrics_all[x] + "_train"
#     xs_test = metrics_all[x] + "_test"
#     fig = plt.figure(figsize=[12, 6])

#     xx = models["Model"]
#     plt.tick_params(labelsize=14)
#     plt.plot(xx, models[xs_train], marker="o", label=xs_train)
#     plt.plot(xx, models[xs_test], marker="o", label=xs_test)
#     plt.legend(prop={"size": 15})
#     plt.title(
#         str(xs) + " of " + str(num_models) + " popular models on train and test data",
#         fontsize=15,
#     )
#     plt.xlabel("Models", fontsize=20)
#     plt.ylabel(xs + ", %", fontsize=20)
#     plt.xticks(xx, rotation=25, fontsize=15)
#     # store the plots
#     Im_dir_path = "/home/zhendi/pm/scripts/image/"
#     # clean_name = re.split('[()]', str(estimator))[0]
#     Name_Formatted = ("%s" % metrics_all[x]) + "_8models_ba.png"
#     file_path = os.path.join(Im_dir_path, Name_Formatted)
#     fig.savefig(file_path)

#     plt.show()
