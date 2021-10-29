# coding: utf-8
import os


__version__ = "0.1.1"


LIB_DIR = os.path.dirname(os.path.abspath(__file__))
USER_HOME_DIR = os.path.expanduser('~')
FILES_DIR = os.path.join(LIB_DIR, 'files/')
PROJECT_HOME_DIR = os.path.join(USER_HOME_DIR, '.fastplm/')
TMP_DIR = '/tmp/'


MODEL_CONFIG_FILE = {
    'google_bert_base_en': os.path.join(FILES_DIR, 'google_bert_base_en.json'),
    'google_bert_base_zh': os.path.join(FILES_DIR, 'google_bert_base_zh.json'),
    'google_albert_base_zh': os.path.join(FILES_DIR, 'google_albert_base_zh.json'),
    'google_albert_base_en': os.path.join(FILES_DIR, 'google_albert_base_en.json'),
    'uer_roberta_base_zh': os.path.join(FILES_DIR, 'uer_roberta_base_zh.json'),
    'uer_roberta_base_en': os.path.join(FILES_DIR, 'uer_roberta_base_en.json'),
    'uer_bert_large_zh': os.path.join(FILES_DIR, 'uer_bert_large_zh.json'),
    'uer_bert_small_zh': os.path.join(FILES_DIR, 'uer_bert_small_zh.json'),
    'uer_bert_tiny_zh': os.path.join(FILES_DIR, 'uer_bert_tiny_zh.json'),
    'uer_gpt_zh': os.path.join(FILES_DIR, 'uer_gpt_zh.json'),
    'uer_gpt_en': os.path.join(FILES_DIR, 'uer_gpt_en.json'),
    'uer_gcnn_9_zh': os.path.join(FILES_DIR, 'uer_gcnn_9_zh.json'),
    'uer_gcnn_9_en': os.path.join(FILES_DIR, 'uer_gcnn_9_en.json'),
    'uer_tower_tiny_zh': os.path.join(FILES_DIR, 'uer_tower_tiny_zh.json'),
    'huawei_tinybert_4_en': os.path.join(FILES_DIR, 'huawei_tinybert_4_en.json'),
    'huawei_tinybert_6_en': os.path.join(FILES_DIR, 'huawei_tinybert_6_en.json'),
}


SINGLE_SENTENCE_CLS_KERNEL_MAP = {
    'google_bert_base_en': 'FastBERT',
    'google_bert_base_zh': 'FastBERT',
    'google_albert_base_zh': 'FastALBERT',
    'google_albert_base_en': 'FastALBERT',
    'uer_bert_large_zh': 'FastBERT',
    'uer_bert_small_zh': 'FastBERT',
    'uer_bert_tiny_zh': 'FastBERT',
    'uer_roberta_base_zh': 'FastBERT',
    'uer_roberta_base_en': 'FastBERT',
    'uer_gpt_zh': 'FastGPT',
    'uer_gpt_en': 'FastGPT',
    'uer_gcnn_9_zh': 'FastGCNN',
    'uer_gcnn_9_en': 'FastGCNN',
    'uer_tower_tiny_zh': 'FastBERT',
    'huawei_tinybert_4_en': 'FastBERT',
    'huawei_tinybert_6_en': 'FastBERT',
}


DOUBLE_SENTENCE_CLS_KERNEL_MAP = {
    'google_bert_base_en': 'FastBERT_S2',
    'google_bert_base_zh': 'FastBERT_S2',
    'google_albert_base_zh': 'FastALBERT_S2',
    'google_albert_base_en': 'FastALBERT_S2',
    'uer_bert_large_zh': 'FastBERT_S2',
    'uer_bert_small_zh': 'FastBERT_S2',
    'uer_bert_tiny_zh': 'FastBERT_S2',
    'uer_roberta_base_zh': 'FastBERT_S2',
    'uer_roberta_base_en': 'FastBERT_S2',
    'uer_gpt_zh': 'FastGPT_S2',
    'uer_gpt_en': 'FastGPT_S2',
    'uer_gcnn_9_zh': 'FastGCNN_S2',
    'uer_gcnn_9_en': 'FastGCNN_S2',
    'uer_tower_tiny_zh': 'FastBERT_S2',
    'huawei_tinybert_4_en': 'FastBERT_S2',
    'huawei_tinybert_6_en': 'FastBERT_S2',
}


DEFAULT_SEQ_LENGTH = 128  # Default sentence length.
DEFAULT_DEVICE = 'cpu'  # Default device.


