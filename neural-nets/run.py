import argparse
import os
import sys
import time
from importlib import import_module

import numpy as np
import pandas as pd
import pickle5 as pickle
import torch

from train_eval import training
from utils import (
    build_dataset,
    get_time_dif,
    plot_acc_graph,
    plot_CE_graph,
    plot_f1score_graph,
    plot_logloss_graph,
)

parser = argparse.ArgumentParser(description="Mulitclass Text Classification")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="choose a model: CNN, BiLSTM",  # RCNN, RNN_Attention, BERT is coming soon
)
parser.add_argument(
    "--cuda", type=int, default=0, help="choose a cuda from: [0,1,2,3,4,5,6,7]"
)
parser.add_argument("--device", type=str, default="cpu", help="choose cuda or cpu")
parser.add_argument("--visdom", type=bool, default=False, help="use visdom or not")
args = parser.parse_args()


if __name__ == "__main__":
    # new parameters
    dataset = "./data/result_non_split_strict.pkl"
    embedding_path = "./data/pretrained_pubmed400D_for_TEXTCNN.pkl"
    with (open(embedding_path, "rb")) as openfile:
        embedding = pickle.load(openfile)
    model_name = args.model
    cuda_num = args.cuda
    device = args.device
    visdom = args.visdom

    # pass to config
    x = import_module("models." + model_name)
    config = x.Config(dataset, embedding)
    config.visdom = visdom
    config.device = torch.device(device)
    config.cuda_num = cuda_num
    if config.device == "cuda":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.cuda.set_device(config.cuda_num)

    # set seeds so the results are the same
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # build data loaders
    print("Loading data...")
    start_time = time.time()
    class_wts, train_loader, valid_loader = build_dataset(config)
    config.class_wts = class_wts
    time_dif = get_time_dif(start_time)
    print("Time usage for loading data:", time_dif)

    # import models
    model = x.Model(config).to(config.device)
    print(model.parameters)

    if config.visdom == True:
        import visdom

        from visualize import Visualizer

        tfmt = "%m%d_%H%M%S"
        vis = Visualizer(time.strftime(tfmt))

    # training
    train_loss, valid_loss, val_accs, val_loglosses, val_f1scores = training(
        config, model, train_loader, valid_loader
    )

    # plots
    plot_logloss_graph(model, val_loglosses)
    plot_acc_graph(model, val_accs)
    plot_f1score_graph(model, val_f1scores)
    plot_CE_graph(model, train_loss, valid_loss)
