# FastPLM

An Empirical Study on Adaptive Inference for Pre-trained Language Model.


## Model Zoo

FastPLM is supported by the [UER](https://github.com/dbiir/UER-py) project, and all of UER high-quality models can be accelerated in the Adaptive inference way.

``FastPLM`` object supports the following models:

|Models (kernel_name)  |URL                               |Description                                               |
|----------------------|----------------------------------|----------------------------------------------------------|
|google_bert_base_en   |https://share.weiyun.com/fpdOtcmz | Google pretrained English BERT-base model on Wiki corpus.|
|google_bert_base_zh   |https://share.weiyun.com/AykBph9V | Google pretrained Chinese BERT-base model on Wiki corpus.|
|uer_bert_large_zh     |https://share.weiyun.com/chx2VhGk | UER pretrained Chinese BERT-large model on mixed corpus. |
|uer_bert_small_zh     |https://share.weiyun.com/wZuVBM5g | UER pretrained Chinese BERT-small model on mixed corpus. |
|uer_bert_tiny_zh      |https://share.weiyun.com/VJ3JEN9Z | UER pretrained Chinese BERT-tiny model on mixed corpus.  |
|uer_roberta_base_zh   |https://share.weiyun.com/2gdpc4P0 | Facebook pretrained Chinese RoBerta-base model.          |
|uer_roberta_base_en   |https://share.weiyun.com/xHrPjgEK | Facebook pretrained English RoBerta-base model.          |
|uer_gpt_zh            |https://share.weiyun.com/Pzn5Iob2 | UER pretrained Chinese GPT model on mixed corpus.        |
|uer_gpt_en            |https://share.weiyun.com/Kc7KlgBs | UER pretrained English GPT model on Wikien corpus.       |
|uer_gcnn_9_zh         |https://share.weiyun.com/AIL6xiPa | UER pretrained Chinese GCNN model on Wiki corpus.        |
|uer_gcnn_9_en         |https://share.weiyun.com/TzjugXLH | UER pretrained English GCNN model on Wikien corpus.      |
|google_albert_base_en |https://share.weiyun.com/artkxjB1 | Google pretrained English ALBERT-base model.             |
|google_albert_base_zh |https://share.weiyun.com/ewaiqKdR | Google pretrained Chinese ALBERT-base model.             |
|huawei_tinybert_4_en  |https://share.weiyun.com/HaulZbd2 | Huawei pretrained English TinyBERT_4 model.              |
|huawei_tinybert_6_en  |https://share.weiyun.com/jPg6FkwA | Huawei pretrained English TinyBERT_6 model.              |


In fact, you don't have to download the model yourself. FastPLM will download the corresponding model file automatically at the first time you use it. 
If the automatically downloading failed, you can download these model files from the above URLs, and saving them to the directory of "~/.fastplm/".


## Quick Start

### Single-sentence classification

An example of single-sentence classification is shown in [single_sentence_classification](examples/single_sentence_classification/).

```python
from fastplm import FastPLM

# Loading your dataset
labels = ['T', 'F']
sents_train = [
    'Do you like FastPLM?',
    'Yes, it runs faster than original PLM!',
    ...
]
labels_train = [
    'T',
    'F',
    ...
]

# Creating a model
model = FastPLM(
    kernel_name="google_bert_base_en",  # "google_bert_base_zh" for Chinese
    labels=labels,
    device='cuda:0'
)

# Training the model
model.fit(
    sents_train,
    labels_train,
    model_saving_path='./fastplm.bin',
)

# Loading the model and making inference
model.load_model('./fastplm.bin')
label, exec_layers = model('I like FastPLM', speed=0.7)
```

### Double-sentence classification

An example of double-sentence classification is shown in [double_sentence_classification](examples/double_sentence_classification/).


## Acknowledgement

This work is supported by Tencent Rhino-Bird Program. A portion of this work was presented at the 58th Annual Meeting of the Association for Computational Linguistics (ACL) in July 2020, [FastBERT](https://github.com/autoliuweijie/FastBERT).

```
@article{weijieliu2021fastplm,
  title={An Empirical Study on Adaptive Inference for Pre-trained Language Model},
  author={Weijie Liu, Xin Zhao, Zhe Zhao, Qi Ju*, Xuefeng Yang, and Wei Lu},
  journal={IEEE TRANSACTIONS ON NEURAL NETWORKS AND LEARNING SYSTEMS},
  year={2021}
}
```
