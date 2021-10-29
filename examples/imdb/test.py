# coding: utf-8
"""
An example of using fastbert model for single sentence classificaion

@author: weijie liu
"""
import os, sys
fastplm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastplm_dir)
import torch
import numpy as np
from fastplm import FastPLM
from train import loading_dataset, kernel_name, model_saving_path


test_dataset_path = "../../datasets/imdb/dev.tsv"


def main():

    sents_test, labels_test = loading_dataset(test_dataset_path)
    samples_num = len(sents_test)
    labels = ["0", "1"]

    model = FastPLM(
        kernel_name=kernel_name,
        labels=labels,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        is_load=False,
        seq_length=256,
    )

    model.load_model(model_saving_path)

    for speed in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        correct_num = 0
        exec_layer_list = []
        for i, (sent, label) in enumerate(zip(sents_test, labels_test)):
            label_pred, exec_layer = model(sent, speed=speed)
            if label_pred == label:
                correct_num += 1
            exec_layer_list.append(exec_layer)

        acc = correct_num / samples_num
        ave_exec_layers = np.mean(exec_layer_list)
        print("Speed = {}, Acc = {:.3f}, Ave_exec_layers = {}".format(speed, acc, ave_exec_layers))


if __name__ == "__main__":
    main()

