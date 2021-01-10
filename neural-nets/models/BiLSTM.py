import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Config(object):

    """parameters"""

    def __init__(self, dataset, embedding):
        # information
        self.model_name = "BiLSTM"  # model name
        self.dataset_path = dataset  # data path
        self.saved_model_name = "BiSTLM_best99"  # for saving figures and pickles
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
        self.dropout2 = 0.5  # dropout
        self.dropout1 = 0.8
        self.n_epochs_stop = 200  # for early stopping
        self.n_classes = len(self.le_classes)
        self.n_vocab = self.embedding_pretrained.shape[0]  # number of vocabulary
        self.num_epochs = 1000
        self.batch_size = 128
        self.learning_rate = 1e-4
        self.embed_size = self.embedding_pretrained.shape[1]
        # self.filter_sizes = [1, 4, 4, 4]
        # self.num_filters = 128
        self.hidden_size1 = 128
        self.hidden_size2 = 256
        self.num_layers = 1


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed_size)
        self.embedding.weight = nn.Parameter(
            torch.tensor(config.embedding_pretrained, dtype=torch.float32)
        )  # weights: [vocab_size, emb_size]
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            config.embed_size,
            config.hidden_size1,
            config.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=config.dropout1,
        )
        self.fc1 = nn.Linear(
            config.hidden_size1 * 4, config.hidden_size2
        )  # weights: [hidden_size * 4, hidden_size]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout2)
        self.fc2 = nn.Linear(
            config.hidden_size2, config.n_classes
        )  # weights: [hidden_size, classes]

    def forward(self, x):

        h_embedding = self.embedding(x)  # [batch_size, feature_length, emb_size]
        # _embedding = torch.squeeze(torch.unsqueeze(h_embedding, 0))
        h_lstm, _ = self.lstm(
            h_embedding
        )  # [batch_size, feature_length, hidden_size*2]
        avg_pool = torch.mean(h_lstm, 1)  # [batch_size, hidden_size * 2]
        max_pool, _ = torch.max(h_lstm, 1)  # [batch_size, hidden_size * 2]
        conc = torch.cat((avg_pool, max_pool), 1)  # [batch_size, hidden_size * 4]

        conc = self.relu(self.fc1(conc))  # [batch_size, hidden_size]
        conc = self.dropout(conc)  # [batch_size, hidden_size]
        out = self.fc2(conc)  # [batch_size, classes]

        return out
