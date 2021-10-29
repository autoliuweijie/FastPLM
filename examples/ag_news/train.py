# coding: utf-8
"""
An example of training single sentence classification model with
douban_book_review dataset.

@author: Weijie Liu
"""
import os, sys
fastplm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastplm_dir)
import torch
from fastplm import FastPLM

kernel_name = 'uer_gcnn_9_en'
seq_length = 128  # 除了gpt, 其他都用256
train_dataset_path = "../../datasets/ag_news/train.tsv"
dev_dataset_path = "../../datasets/ag_news/test.tsv"
model_saving_path = "./{}_ag_news.bin".format(kernel_name)
is_load=True

def loading_dataset(dataset_path):
    sents, labels = [], []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i == 0:
                continue
            line = line.strip().split('\t')
            sents.append(line[1])
            labels.append(line[0])
    return sents, labels


def main():

    sents_train, labels_train = loading_dataset(train_dataset_path)
    sents_dev, labels_dev = loading_dataset(dev_dataset_path)
    labels = ["1", "2", "3", "4"]
    print("Labels: ", labels)  # [0, 1]

    # Create Model
    model = FastPLM(
        kernel_name=kernel_name,
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        is_load=is_load,
        seq_length=seq_length
    )

    model.fit(
        sents_train,
        labels_train,
        sentences_dev=sents_dev,
        labels_dev=labels_dev,
        finetuning_epochs_num=3,
        distilling_epochs_num=5,
        report_steps=100,
        model_saving_path=model_saving_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
