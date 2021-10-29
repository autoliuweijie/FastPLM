# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from .config import *
from .utils import *
from .fastbert import FastBERT, FastBERT_S2
from .classifiers import DenseMiniClassifier


class FastGPT(FastBERT):

    MiniClassifier = DenseMiniClassifier

    def _mask_transfer(self,
                       mask, # batch_size x seq_length
                       emb):
        batch_size, seq_length, emb_size = emb.size()
        mask = torch.ones(seq_length, seq_length, device=emb.device)
        mask = torch.tril(mask)
        mask = (1.0 - mask) * -10000.0
        mask = mask.repeat(batch_size, 1, 1, 1)
        return mask


class FastGPT_S2(FastBERT_S2, FastGPT):

    MiniClassifier = FastGPT.MiniClassifier

    def _mask_transfer(self,
                       mask,
                       emb):
        return FastGPT._mask_transfer(self, mask, emb)


