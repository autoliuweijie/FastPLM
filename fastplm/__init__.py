# coding:utf-8
import os, sys
from .config import *


sys.path.append(LIB_DIR)
if not os.path.exists(PROJECT_HOME_DIR):
    os.mkdir(PROJECT_HOME_DIR)


from .fastbert import FastBERT
from .fastbert import FastBERT_S2
from .fastgpt import FastGPT
from .fastgpt import FastGPT_S2
from .fastgcnn import FastGCNN
from .fastgcnn import FastGCNN_S2
from .fastalbert import FastALBERT
from .fastalbert import FastALBERT_S2

def FastPLM(kernel_name,
              **kwargs):
    model = globals()[SINGLE_SENTENCE_CLS_KERNEL_MAP[kernel_name]](kernel_name, **kwargs)
    return model


def FastPLM_S2(kernel_name,
                 **kwargs):
    model = globals()[DOUBLE_SENTENCE_CLS_KERNEL_MAP[kernel_name]](kernel_name, **kwargs)
    return model

