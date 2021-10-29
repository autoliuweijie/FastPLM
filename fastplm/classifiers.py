# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from .config import *
from .utils import *
from .uer.layers.multi_headed_attn import MultiHeadedAttention


class MiniClassifier(nn.Module):

    def __init__(self,
                 args,
                 input_size,
                 labels_num):
        super(MiniClassifier, self).__init__()
        self.args = args
        self.input_size = input_size
        self.labels_num = labels_num

    def forward(self,
                hidden,
                mask):
        pass


class DenseMiniClassifier(MiniClassifier):

    def __init__(self,
                 args,
                 input_size,
                 labels_num):
        super(DenseMiniClassifier, self).__init__(args, input_size, labels_num)
        self.pooling = args.pooling
        self.output_layer_1 = nn.Linear(input_size, input_size)
        self.output_layer_2 = nn.Linear(input_size, labels_num)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,
                hidden,
                mask):
        output = hidden
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]
        output = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output)
        return logits


class AttMiniClassifier(MiniClassifier):

    def __init__(self,
                 args,
                 input_size,
                 labels_num):
        super(AttMiniClassifier, self).__init__(args, input_size, labels_num)
        self.cla_hidden_size = 128
        self.cla_heads_num = 2
        self.pooling = args.pooling
        self.output_layer_0 = nn.Linear(input_size, self.cla_hidden_size)
        self.self_atten = MultiHeadedAttention(self.cla_hidden_size, self.cla_heads_num, args.dropout)
        self.output_layer_1 = nn.Linear(self.cla_hidden_size, self.cla_hidden_size)
        self.output_layer_2 = nn.Linear(self.cla_hidden_size, labels_num)

    def forward(self,
                hidden,
                mask):

        hidden = torch.tanh(self.output_layer_0(hidden))
        hidden = self.self_atten(hidden, hidden, hidden, mask)

        if self.pooling == "mean":
            hidden = torch.mean(hidden, dim=1)
        elif self.pooling == "max":
            hidden = torch.max(hidden, dim=1)[0]
        elif self.pooling == "last":
            hidden = hidden[:, -1, :]
        else:
            hidden = hidden[:, 0, :]

        output_1 = torch.tanh(self.output_layer_1(hidden))
        logits = self.output_layer_2(output_1)
        return logits
