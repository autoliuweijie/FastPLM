# coding: utf-8
"""
An example of training two sentences classification model with
QNLI dataset.

@author: Weijie Liu
"""
import os
import torch
import sys
fastplm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(fastplm_dir)
from fastplm import FastPLM_S2

kernel_name = 'google_albert_base_en'
train_dataset_path = "../../datasets/datasets/QQP/qqp.train.tsv"
dev_dataset_path = "../../datasets/datasets/QQP/qqp.dev.tsv"
model_saving_path = "/tmp/{}_qqp.bin".format(kernel_name)
is_load=True
labels = ['0', '1']


def loading_dataset(dataset_path):
    sents_a, sents_b, labels = [], [], []
    with open(dataset_path, 'r', encoding='utf-8') as infile:
        for i, line in enumerate(infile):
            if i == 0:
                continue
            line = line.strip().split('\t')
            sents_a.append(line[0])
            sents_b.append(line[1])
            labels.append(line[2])
    return sents_a, sents_b, labels


def main():

    sents_a_train, sents_b_train, labels_train = loading_dataset(train_dataset_path)
    sents_a_dev, sents_b_dev, labels_dev = loading_dataset(dev_dataset_path)
    print("Labels: ", labels)  # [0, 1]

    # Create model
    model = FastPLM_S2(
        kernel_name=kernel_name,
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        # is_load=False,
    )

    model.fit(
        sents_a_train=sents_a_train,
        sents_b_train=sents_b_train,
        labels_train=labels_train,
        sents_a_dev=sents_a_dev,
        sents_b_dev=sents_b_dev,
        labels_dev=labels_dev,
        batch_size=32,
        seq_length=256,
        finetuning_epochs_num=3,
        distilling_epochs_num=10,
        learning_rate=2e-5,
        report_steps=500,
        model_saving_path=model_saving_path,
        verbose=True,
    )


if __name__ == "__main__":
    main()
