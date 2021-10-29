# coding=utf-8
import os
from sklearn.metrics import f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
import numpy as np


GLUE_TASKS = {
        "cola":   {'type': 'single', 'labels_num': 2, 'name': 'CoLA'},  # 单句二分类，判断语法是否正确
        "sst-2":  {'type': 'single', 'labels_num': 2, 'name': 'SST-2'}, # 单句二分类，判断是正负情绪
        "mrpc":   {'type': 'double', 'labels_num': 2, 'name': 'MRPC'},  # 双句二分类，判端是否同语义
        "sts-b":  {'type': 'double', 'labels_num': 5, 'name': 'STS-B'}, # 双句五分类, 判断1-5相似度得分
        "qqp":    {'type': 'double', 'labels_num': 2, 'name': 'QQP'},   # 双句二分类，判断两个问句是否等效
        "mnli":   {'type': 'double', 'labels_num': 3, 'name': 'MNLI'},  # 双句三分类, 判断蕴含、矛盾、中立, 训练测试数据源一致
        "mnli-mm":{'type': 'double', 'labels_num': 3, 'name': 'MNLI'},  # 双句三分类, 判断蕴含、矛盾、中立, 训练测试数据源不一致
        "qnli":   {'type': 'double', 'labels_num': 2, 'name': 'QNLI'},  # 双句二分类, 判断是否蕴含
        "rte":    {'type': 'double', 'labels_num': 2, 'name': 'RTE'},   # 双句二分类，判断是否蕴含
        "wnli":   {'type': 'double', 'labels_num': 2, 'name': 'WNLI'},  # 双句二分类，判断是否相关
    }


def simple_accuracy(preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def glue_compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == "cola":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif task_name == "sst-2":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "mrpc":
        return acc_and_f1(preds, labels)
    elif task_name == "sts-b":
        return pearson_and_spearman(preds, labels)
    elif task_name == "qqp":
        return acc_and_f1(preds, labels)
    elif task_name == "mnli":
        return {"mnli/acc": simple_accuracy(preds, labels)}
    elif task_name == "mnli-mm":
        return {"mnli-mm/acc": simple_accuracy(preds, labels)}
    elif task_name == "qnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "rte":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "wnli":
        return {"acc": simple_accuracy(preds, labels)}
    elif task_name == "hans":
        return {"acc": simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def load_dataset_by_task_name(args, task):
    task_dir_name = GLUE_TASKS[task.lower()]['name']  # 转换成标准名称
    data_dir = os.path.join(args.glue_dir, task_dir_name)
    print("Loading dataset from {}".format(data_dir))

    if task == "cola":
        tests, devs, trains = {'text': [], 'label': []}, {'text': [], 'label': []}, {'text': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                if filename == 'test.tsv':
                    for i, line in enumerate(datafile):
                        if i == 0: continue
                        items = line.strip().split('\t')
                        tmps['text'].append(items[1])
                        tmps['label'].append('unk')
                elif filename in ['dev.tsv', 'train.tsv']:
                    for i, line in enumerate(datafile):
                        if i == 0: continue
                        items = line.strip().split('\t')
                        tmps['text'].append(items[3])
                        tmps['label'].append(int(items[1]))
        return tests, devs, trains

    elif task == "sst-2":
        tests, devs, trains = {'text': [], 'label': []}, {'text': [], 'label': []}, {'text': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                if filename == 'test.tsv':
                    for i, line in enumerate(datafile):
                        if i == 0: continue
                        items = line.strip().split('\t')
                        tmps['text'].append(items[1])
                        tmps['label'].append('unk')
                elif filename in ['dev.tsv', 'train.tsv']:
                    for i, line in enumerate(datafile):
                        if i == 0: continue
                        items = line.strip().split('\t')
                        tmps['text'].append(items[0])
                        tmps['label'].append(int(items[1]))
        return tests, devs, trains

    elif task == "mrpc":
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([devs, tests, trains], ['dev.tsv', 'test.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    tmps['text_a'].append(items[3])
                    tmps['text_b'].append(items[4])
                    if filename == 'test.tsv':
                        tmps['label'].append('unk')
                    else:
                        tmps['label'].append(int(items[0]))
        return tests, devs, trains

    elif task == "sts-b":
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': [], 'score': []}, \
                {'text_a': [], 'text_b': [], 'label': [], 'score': []}, \
                {'text_a': [], 'text_b': [], 'label': [], 'score': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    tmps['text_a'].append(items[7])
                    tmps['text_b'].append(items[8])
                    if filename == 'test.tsv':
                        tmps['label'].append('unk')
                        tmps['score'].append(-1.0)
                    else:
                        score = float(items[9])
                        label = int(score)
                        label = label if label < 5 else 4
                        tmps['label'].append(label)
                        tmps['score'].append(score)
        return tests, devs, trains

    elif task == 'qqp':
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    if filename == 'test.tsv':
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append('unk')
                    else:
                        if len(items) != 6: continue
                        tmps['text_a'].append(items[3])
                        tmps['text_b'].append(items[4])
                        tmps['label'].append(int(items[5]))
        return tests, devs, trains

    elif task == "mnli":
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test_matched.tsv', 'dev_matched.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    tmps['text_a'].append(items[8])
                    tmps['text_b'].append(items[9])
                    if filename == 'test_matched.tsv':
                        tmps['label'].append('unk')
                    elif filename == 'dev_matched.tsv':
                        tmps['label'].append(label_map[items[15]])
                    elif filename == 'train.tsv':
                        tmps['label'].append(label_map[items[10]])
        return tests, devs, trains

    elif task == "mnli-mm":
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test_mismatched.tsv', 'dev_mismatched.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    tmps['text_a'].append(items[8])
                    tmps['text_b'].append(items[9])
                    if filename == 'test_mismatched.tsv':
                        tmps['label'].append('unk')
                    elif filename == 'dev_mismatched.tsv':
                        tmps['label'].append(label_map[items[15]])
                    elif filename == 'train.tsv':
                        tmps['label'].append(label_map[items[10]])
        return tests, devs, trains

    elif task == "qnli":
        label_map = {'entailment': 0, 'not_entailment': 1}
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    if filename == 'test.tsv':
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append('unk')
                    else:
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append(label_map[items[3]])
        return tests, devs, trains

    elif task == "rte":
        label_map = {'entailment': 0, 'not_entailment': 1}
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    if filename == 'test.tsv':
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append('unk')
                    else:
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append(label_map[items[3]])
        return tests, devs, trains

    elif task == "wnli":
        tests, devs, trains = {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}, \
                {'text_a': [], 'text_b': [], 'label': []}
        for tmps, filename in zip([tests, devs, trains], ['test.tsv', 'dev.tsv', 'train.tsv']):
            with open(os.path.join(data_dir, filename), 'r', encoding='utf-8') as datafile:
                for i, line in enumerate(datafile):
                    if i == 0: continue
                    items = line.strip().split('\t')
                    if filename == 'test.tsv':
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append('unk')
                    else:
                        tmps['text_a'].append(items[1])
                        tmps['text_b'].append(items[2])
                        tmps['label'].append(int(items[3]))
        return tests, devs, trains


