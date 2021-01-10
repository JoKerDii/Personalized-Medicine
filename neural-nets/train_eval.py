import os
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchnet as tnt
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics.classification import log_loss

from utils import (
    get_time_dif,
    plot_acc_graph,
    plot_CE_graph,
    plot_confusion_matrix,
    plot_f1score_graph,
    plot_logloss_graph,
)

# from visualize import Visualizer


def training(config, model, train_loader, valid_loader):

    # convert class weights to tensor
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_wts, dtype=torch.float).to(config.device)
    )

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate
    )  # 0.001 0.00001

    ## CNN
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.00001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        verbose=False,
    )
    ## BiLSTM
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     div_factor=250,
    #     max_lr=0.1,
    #     steps_per_epoch=len(train_loader),
    #     epochs=10,
    #     anneal_strategy="linear",
    # )

    # Returns lists of metrics
    train_loss = []  # saving training CE loss
    valid_loss = []  # saving testing CE loss
    val_accs = []  # saving accuracy
    val_loglosses = []  # saving log loss
    val_f1scores = []  # saving f1 score

    # For early stopping
    min_val_loss = np.Inf
    max_val_acc = -1
    epochs_no_improve = 0
    # n_epochs_stop = 6
    early_stop = False

    print("Training...")
    start_time = time.time()

    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))

        model.train()
        avg_tr_loss = tnt.meter.AverageValueMeter()
        total_tr_preds = np.empty(shape=(0, 9), dtype=int)
        for i, batch in enumerate(train_loader):
            batch = [r.to(config.device) for r in batch]
            x_batch, y_batch = batch
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred = F.softmax(y_pred, dim=1).detach().cpu().numpy()  # (batch_size, 9)
            total_tr_preds = np.append(
                total_tr_preds, y_pred, axis=0
            )  # (batch_size, 9)
            avg_tr_loss.add(loss.item())

        scheduler.step(avg_tr_loss.value()[0])

        avg_ts_loss, val_acc, val_log_loss, val_f1score = evaluate(
            config, model, valid_loader
        )

        ### print out ###
        print(f"Epoch: {epoch + 1}")
        print(
            f"\tCELoss: {avg_tr_loss.value()[0]:.4f}(train)\t|\tCELoss: {avg_ts_loss:.4f}(valid)"
        )
        print(
            f"\tLogLoss: {val_log_loss :.4f}(valid)\t|\tAcc: {val_acc * 100:.2f}%(valid)\t|\tF1score: {val_f1score:.4f}(valid)"
        )

        ### vis ###
        if config.visdom == True:
            import visdom

            from visualize import Visualizer

            vis.text(
                f"Epoch: {epoch + 1} \n \tCELoss: {avg_tr_loss:.4f}(train)\t|\tCELoss: {avg_ts_loss:.4f}(valid) \n \tLogLoss: {val_log_loss :.4f}(valid)\t|\tAcc: {val_acc * 100:.2f}%(valid)\t|\tF1score: {val_f1score:.4f}(valid)",
                win="CNN",
            )

            vis.plot("Training CE loss", avg_tr_loss)
            vis.plot("Testing CE loss", avg_ts_loss)
            vis.plot("accuracy", val_acc)
            vis.plot("log_loss", val_log_loss)
            vis.plot("f1score", val_f1score)

        ### save ###
        # CE loss, validation accuracy, log loss, f1 score
        train_loss.append(avg_tr_loss.value()[0])
        valid_loss.append(avg_ts_loss)
        val_accs.append(val_acc)
        val_loglosses.append(val_log_loss)
        val_f1scores.append(val_f1score)

        if avg_ts_loss < min_val_loss:
            epochs_no_improve = 0
            min_val_loss = avg_ts_loss
        else:
            epochs_no_improve += 1

        if val_acc > max_val_acc:
            # save the model
            if config.model_name != None:
                torch.save(model.state_dict(), config.save_path)
            max_val_acc = val_acc

        # early stopping
        if config.allow_early_stop == True:
            if epoch > 10 and epochs_no_improve == config.n_epochs_stop:
                print("Holy Shit!")
                early_stop = True

            if early_stop:
                print("Fucking Stopped")
                break

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # plots CE, acc, logloss, and f1score curves
    plot_CE_graph(config, train_loss, valid_loss)
    plot_acc_graph(config, val_accs)
    plot_logloss_graph(config, val_loglosses)
    plot_f1score_graph(config, val_f1scores)

    # using current model weights on all validation set
    final_test(config, model, valid_loader)
    return train_loss, valid_loss, val_accs, val_loglosses, val_f1scores


def final_test(config, model, valid_loader):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    (
        total_ts_labels,  # for confusion matrix
        total_ts_preds,  # for confusion matrix
        encoded_ts_labels,
        avg_ts_loss,
        val_acc,  # single value, average testing accuracy
        val_log_loss,  # single value, average testing log loss
        val_f1score,  # single value , average testing f1 score
    ) = evaluate(config, model, valid_loader, test=True)
    print("Final Test...")
    print("Log loss: {}".format(val_log_loss))
    print("Accuracy: {}".format(val_acc))
    print("F1 score: {}".format(val_f1score))

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # plot confusion matrix
    plot_confusion_matrix(config, total_ts_labels, total_ts_preds.argmax(axis=1))


def evaluate(config, model, valid_loader, test=False):
    model.eval()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_wts, dtype=torch.float).to(config.device)
    )
    total_ts_labels = np.array([], dtype=int)
    total_ts_preds = np.empty(shape=(0, 9), dtype=int)

    avg_ts_loss = tnt.meter.AverageValueMeter()
    with torch.no_grad():
        for i, batch in enumerate(valid_loader):
            batch = [r.to(config.device) for r in batch]
            x_batch, y_batch = batch
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            avg_ts_loss.add(loss.item())
            y_pred = F.softmax(y_pred, dim=1).detach().cpu().numpy()  # (batch_size, 9)
            total_ts_labels = np.append(
                total_ts_labels, y_batch.cpu().numpy()
            )  # (batch_size, 1)
            total_ts_preds = np.append(
                total_ts_preds, y_pred, axis=0
            )  # (batch_size, 9)

    encoded_ts_labels = pd.get_dummies(total_ts_labels)  # (N, 9)

    # Accuracy
    val_acc = balanced_accuracy_score(
        total_ts_labels, total_ts_preds.argmax(axis=1)
    )  # argmax -> numeric labels (batch_size, 1)
    # log loss
    val_log_loss = log_loss(encoded_ts_labels, total_ts_preds)
    # f1 score
    val_f1score = f1_score(
        total_ts_labels, total_ts_preds.argmax(axis=1), average="micro"
    )

    if test == True:
        return (
            total_ts_labels,
            total_ts_preds,
            encoded_ts_labels,
            avg_ts_loss.value()[0],
            val_acc,
            val_log_loss,
            val_f1score,
        )
    else:
        return avg_ts_loss.value()[0], val_acc, val_log_loss, val_f1score
