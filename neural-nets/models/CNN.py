import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """parameters"""

    def __init__(self, dataset, embedding):
        # information
        self.model_name = "CNN"  # model name
        self.dataset_path = dataset  # data path
        self.saved_model_name = "CNN_best99"  # for saving figures and pickles
        self.save_path = (
            "./model_ckpt/" + self.saved_model_name + ".ckpt"
        )  # saved model path
        self.log_path = "./model_log/" + self.saved_model_name
        self.im_cm_path = "./image/cm/" + self.saved_model_name + ".png"
        self.im_CE_path = "./image/CE/" + self.saved_model_name + ".png"
        self.im_acc_path = "./image/acc/" + self.saved_model_name + ".png"
        self.im_logloss_path = "./image/logloss/" + self.saved_model_name + ".png"
        self.im_f1score_path = "./image/f1score/" + self.saved_model_name + ".png"
        self.le_classes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])  # all classes
        self.max_vocab_size = 10000
        self.allow_early_stop = True
        self.class_wts = None  # weights for cross entropy loss
        self.embedding_pretrained = embedding  # embedding matrix
        self.visdom = False

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("cpu")  # default device is cpu

        # model parameters
        self.dropout = 0.5  # dropout
        self.n_epochs_stop = 200  # for early stopping
        self.n_classes = len(self.le_classes)
        self.n_vocab = self.embedding_pretrained.shape[0]  # number of vocabulary
        self.num_epochs = 1000
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.embed_size = self.embedding_pretrained.shape[1]
        self.filter_sizes = [1, 4, 4, 4]
        self.num_filters = 128


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(config.embedding_pretrained, dtype=torch.float32)
        )  # weights: [vocab_size, emb_size]
        self.embedding.weight.requires_grad = False
        self.convs1 = nn.ModuleList(
            [
                nn.Conv2d(1, config.num_filters, (K, config.embed_size))
                for K in config.filter_sizes
            ]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(
            len(config.filter_sizes) * config.num_filters, config.n_classes
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(
            3
        )  # conv [batch_size, n_filters, feature_length, 1] -> squeeze(3) [batch_size, n_filters, feature_length] -> conv [batch_size, n_filters, feature_length-1, 1] -> squeeze(3) [batch_size, n_filters, feature_length-1] -> conv [batch_size, n_filters, feature_length-2, 1] ......
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # [batch_size, n_filters]
        return x

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, feature_len, emb_size]
        x = x.unsqueeze(1)  # [batch_size, 1, feature_len, emb_size]
        # x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        # x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        # x = torch.cat(x, 1)
        x = torch.cat(
            [self.conv_and_pool(x, conv) for conv in self.convs1], 1
        )  # each [batch_size, n_filters], len(filter_sizes) in total = len(filter_sizes)*num_filters
        x = self.dropout(x)
        logit = self.fc1(x)  # [batch_size, classes]
        return logit
