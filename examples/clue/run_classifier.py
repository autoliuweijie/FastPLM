# coding: utf-8
import os, sys
fastplm_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(fastplm_dir)
import torch
import argparse
import time
import numpy as np
from tqdm import tqdm
from fastplm import FastPLM, FastPLM_S2
from clue_utils import load_dataset_by_task_name
from thop import profile
from thop import clever_format


def build_model(args, task_type, labels):
    assert isinstance(labels, list)
    if task_type == 'single':
        creator = FastPLM
    elif task_type == 'double':
        creator = FastPLM_S2
    print("Building {}-sentence classification model with kernel of {} and labels of {}.".\
            format(task_type, args.kernel, labels))
    model = creator(
            kernel_name=args.kernel,
            labels=labels,
            seq_length=args.seq_length,
            device=args.device,
            pooling=args.pooling,
            strict_length=args.strict_length,
            is_load=args.load_pretrained_model)
    return model


def evaluate(args, model, dataset, subtitle):
    print("Evaluating model on {}-{} task.".format(args.task, subtitle))
    res = []
    for speed in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        start_time = time.time()
        predict_labels, exec_layers = [], []
        if args.task_type == 'single':
            for sent in dataset['text_a']:
                lp, el = model(sent, speed=speed)
                predict_labels.append(lp)
                exec_layers.append(el)
        elif args.task_type == 'double':
            for sent_a, sent_b in zip(dataset['text_a'], dataset['text_b']):
                lp, el = model(sent_a, sent_b, speed=speed)
                predict_labels.append(lp)
                exec_layers.append(el)
        end_time = time.time()
        ave_exec_layer = np.mean(exec_layers)
        cost_time = end_time - start_time

        correct_num = 0
        for tl, pl in zip(dataset['label'], predict_labels):
            if tl == pl:
                correct_num += 1
        accuracy = float(correct_num) / len(predict_labels)

        res_tmp = {}
        res_tmp['speed'] = speed
        res_tmp['accuracy'] = accuracy
        res_tmp['ave_exec_layer'] = ave_exec_layer
        res_tmp['cost_time'] = cost_time
        res_tmp['time_per_sample'] = cost_time/len(dataset['text_a'])
        for k, v in res_tmp.items():
            print("|{}={:4f}".format(k, v), end='')
        print("|")
        res.append(res_tmp)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", type=str, required=True, help="Name of the kernel model.")
    parser.add_argument("--task", type=str, required=True, help="Name of the task.")
    parser.add_argument("--clue_dir", type=str, required=True, help="Path of the CLUE dir.")
    parser.add_argument("--pooling", type=str, default=None, help="first, mean, last, max.")
    parser.add_argument("--do_train", action="store_true", help="Whether training.")
    parser.add_argument("--seq_length", type=int, default=128, help="Length of the sentence.")
    parser.add_argument("--strict_length", action="store_true", help="Strict length to half of seq_length for two sentences classification.")
    parser.add_argument("--load_pretrained_model", action="store_true", help="Whether load pretrained model.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--finetuning_epochs_num", type=int, default=3, help="The number of fintuning epochs.")
    parser.add_argument("--distilling_epochs_num", type=int, default=5, help="The epochs number of self-distillation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--model_saving_path", type=str, default='/tmp/{}_{}.bin', help="Path of the output model.")
    parser.add_argument("--report_steps", type=int, default=100, help="Report step.")
    args = parser.parse_args()

    args.task = args.task.lower()
    args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.model_saving_path = args.model_saving_path.format(args.kernel, args.task)
    print("Using {}.".format(args.device))

    trains, devs, tests, task_type = load_dataset_by_task_name(args.task, args.clue_dir)
    args.task_type = task_type
    print("There are {} samples in training set, and {} samples in developing set.".\
            format(len(trains['label']), len(devs['label'])))

    labels = list(set(trains['label']))
    labels.sort()
    model = build_model(args, task_type, labels)

    # print model size
    # if task_type == 'single':
        # inputs = ['sentences']
    # elif task_type == 'double':
        # inputs = ['sentence A', 'sentence B']
    # flops, params = profile(model, inputs=inputs)
    # print(params)


    # Model training
    if args.do_train and task_type == 'single':
        model.fit(
            trains['text_a'],
            trains['label'],
            sentences_dev=devs['text_a'],
            labels_dev=devs['label'],
            batch_size=args.batch_size,
            learnining_rate=args.learning_rate,
            finetuning_epochs_num=args.finetuning_epochs_num,
            distilling_epochs_num=args.distilling_epochs_num,
            report_steps=args.report_steps,
            model_saving_path=args.model_saving_path,
            verbose=True
        )
    elif args.do_train and task_type == 'double':
        model.fit(
            sents_a_train=trains['text_a'],
            sents_b_train=trains['text_b'],
            labels_train=trains['label'],
            sents_a_dev=devs['text_a'],
            sents_b_dev=devs['text_b'],
            labels_dev=devs['label'],
            batch_size=args.batch_size,
            learnining_rate=args.learning_rate,
            finetuning_epochs_num=args.finetuning_epochs_num,
            distilling_epochs_num=args.distilling_epochs_num,
            report_steps=args.report_steps,
            model_saving_path=args.model_saving_path,
            verbose=True
        )

    # Evaluating
    model.load_model(args.model_saving_path)
    evaluate(args, model, devs, subtitle='dev')


if __name__ == "__main__":
    main()

