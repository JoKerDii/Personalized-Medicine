import os
import re
import time
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle5 as pickle
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight


def build_dataset(config):
    """
    Input is the preprocessed data. Going through spliting, tokenizing, padding, creating loaders
    """
    with (open(config.dataset_path, "rb")) as openfile:
        data = pickle.load(openfile)
    # data = pd.read_pickle(config.dataset_path)
    train_X, test_X, train_y, test_y = train_test_split(
        data["TEXT"],
        data["Class"].values - 1,
        test_size=0.2,
        random_state=5,
        stratify=data["Class"].values - 1,
    )

    class_wts = compute_class_weight("balanced", np.unique(train_y), train_y)
    print("train_X.shape:", train_X.shape)
    print("test_X.shape:", test_X.shape)
    print("train_y.shape:", train_y.shape)
    print("test_y.shape:", test_y.shape)

    # tokenizing
    tokenizer = Tokenizer(num_words=config.n_vocab)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    # padding
    train_X = pad_sequences(train_X, maxlen=config.max_vocab_size)
    test_X = pad_sequences(test_X, maxlen=config.max_vocab_size)

    # labeling
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)  # int64
    test_y = le.transform(test_y)  # test_y must begin from 0

    # to tensor
    x_train = torch.tensor(train_X, dtype=torch.long)
    y_train = torch.tensor(train_y, dtype=torch.long)
    x_cv = torch.tensor(test_X, dtype=torch.long)
    y_cv = torch.tensor(test_y, dtype=torch.long)

    # create torch data
    train = torch.utils.data.TensorDataset(x_train, y_train)
    valid = torch.utils.data.TensorDataset(x_cv, y_cv)

    # create data loaders
    train_loader = torch.utils.data.DataLoader(
        train,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
    )

    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)
    print("x_cv.shape:", x_cv.shape)
    print("y_cv.shape:", y_cv.shape)

    return class_wts, train_loader, valid_loader


def plot_confusion_matrix(config, test_y, predict_y):

    """
    test_y is the true labels, predict_y is the predicted labels
    Both have to be an array or list of numeric integers (not onehot labels)
    Both begins from 0.
    """

    C = confusion_matrix(test_y, predict_y)

    labels = list(config.le_classes + 1)
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(
        C, annot=True, cmap="Blues", fmt=".3f", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted Class", fontsize=15)
    plt.ylabel("Original Class", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    fig.savefig(config.im_cm_path)
    plt.show()


def plot_CE_graph(config, train_loss, valid_loss):

    fig = plt.figure(figsize=(12, 12))
    plt.title("Train/Validation Cross Entropy Loss", fontsize=20)
    plt.plot(list(np.arange(len(train_loss)) + 1), train_loss, label="train")
    plt.plot(list(np.arange(len(train_loss)) + 1), valid_loss, label="validation")
    plt.xlabel("num_epochs", fontsize=18)
    plt.ylabel("loss", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best", fontsize=15)
    fig.savefig(config.im_CE_path)
    plt.show()


def plot_acc_graph(config, val_accs):
    fig = plt.figure(figsize=(12, 12))
    plt.title("Validation Accuracy", fontsize=20)
    plt.plot(list(np.arange(len(val_accs)) + 1), val_accs)
    plt.xlabel("num_epochs", fontsize=18)
    plt.ylabel("accuracy", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best")
    fig.savefig(config.im_acc_path)
    plt.show()


def plot_logloss_graph(config, val_loglosses):
    fig = plt.figure(figsize=(12, 12))
    plt.title("Validation Log Loss", fontsize=20)
    plt.plot(list(np.arange(len(val_loglosses)) + 1), val_loglosses)
    plt.xlabel("num_epochs", fontsize=18)
    plt.ylabel("log loss", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best")
    fig.savefig(config.im_logloss_path)
    plt.show()


def plot_f1score_graph(config, val_f1scores):
    fig = plt.figure(figsize=(12, 12))
    plt.title("Validation F1 Score", fontsize=20)
    plt.plot(list(np.arange(len(val_f1scores)) + 1), val_f1scores)
    plt.xlabel("num_epochs", fontsize=18)
    plt.ylabel("F1 Score", fontsize=18)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc="best")
    fig.savefig(config.im_f1score_path)
    plt.show()


def get_time_dif(start_time):
    """obtain time difference"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    pass
