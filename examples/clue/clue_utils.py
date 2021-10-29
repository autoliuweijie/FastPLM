# coding: utf-8
import os
import json


SINGLE_TYPE_NAME = 'single'
DOUBLE_TYPE_NAME = 'double'


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_type(self):
        """Gets the task type."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r", encoding="utf-8") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines


class TnewsProcessor(DataProcessor):
    """Processor for the TNEWS data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return SINGLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "100"
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class IflytekProcessor(DataProcessor):
    """Processor for the IFLYTEK data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return SINGLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            text_b = None
            label = str(line['label']) if set_type != 'test' else "0"
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class AfqmcProcessor(DataProcessor):
    """Processor for the AFQMC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return DOUBLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence1']
            text_b = line['sentence2']
            label = str(line['label']) if set_type != 'test' else "0"
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class CmnliProcessor(DataProcessor):
    """Processor for the CMNLI data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return DOUBLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line["sentence1"]
            text_b = line["sentence2"]
            label = str(line["label"]) if set_type != 'test' else 'neutral'
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class CslProcessor(DataProcessor):
    """Processor for the CSL data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return DOUBLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = " ".join(line['keyword'])
            text_b = line['abst']
            label = str(line['label']) if set_type != 'test' else '0'
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class WscProcessor(DataProcessor):
    """Processor for the WSC data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return SINGLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['text']
            text_a_list = list(text_a)
            target = line['target']
            query = target['span1_text']
            query_idx = target['span1_index']
            pronoun = target['span2_text']
            pronoun_idx = target['span2_index']
            assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))] == pronoun, "pronoun: {}".format(pronoun)
            assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)
            if pronoun_idx > query_idx:
                text_a_list.insert(query_idx, "_")
                text_a_list.insert(query_idx + len(query) + 1, "_")
                text_a_list.insert(pronoun_idx + 2, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
            else:
                text_a_list.insert(pronoun_idx, "[")
                text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
                text_a_list.insert(query_idx + 2, "_")
                text_a_list.insert(query_idx + len(query) + 2 + 1, "_")
            text_a = "".join(text_a_list)
            text_b = None
            label = str(line['label']) if set_type != 'test' else 'true'
            examples['guid'].append(guid)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)
        return examples


class CopaProcessor(DataProcessor):
    """Processor for the COPA data set (CLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_type(self):
        """See base class."""
        return DOUBLE_TYPE_NAME

    def _create_examples(self, lines, set_type):
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            i = 2 * i
            guid1 = "%s-%s" % (set_type, i)
            guid2 = "%s-%s" % (set_type, i + 1)
            premise = line['premise']
            choice0 = line['choice0']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            choice1 = line['choice1']
            label2 = str(0 if line['label'] == 0 else 1) if set_type != 'test' else '0'
            if line['question'] == 'effect':
                text_a = premise
                text_b = choice0
                text_a2 = premise
                text_b2 = choice1
            elif line['question'] == 'cause':
                text_a = choice0
                text_b = premise
                text_a2 = choice1
                text_b2 = premise
            else:
                raise ValueError(f'unknowed {line["question"]} type')

            examples['guid'].append(guid1)
            examples['text_a'].append(text_a)
            examples['text_b'].append(text_b)
            examples['label'].append(label)

            examples['guid'].append(guid2)
            examples['text_a'].append(text_a2)
            examples['text_b'].append(text_b2)
            examples['label'].append(label2)
        return examples

    def _create_examples_version2(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = {'guid': [], 'text_a': [], 'text_b': [], 'label': []}
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if line['question'] == 'cause':
                text_a = line['premise'] + '这是什么原因造成的？' + line['choice0']
                text_b = line['premise'] + '这是什么原因造成的？' + line['choice1']
            else:
                text_a = line['premise'] + '这造成了什么影响？' + line['choice0']
                text_b = line['premise'] + '这造成了什么影响？' + line['choice1']
            label = str(1 if line['label'] == 0 else 0) if set_type != 'test' else '0'
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


CLUE_TASKS_NUM_LABELS = {
    'iflytek': 119,
    'cmnli': 3,
    'afqmc': 2,
    'csl': 2,
    'wsc': 2,
    'copa': 2,
    'tnews': 15,
}

CLUE_PROCESSORS = {
    'tnews': TnewsProcessor,
    'iflytek': IflytekProcessor,
    'cmnli': CmnliProcessor,
    'afqmc': AfqmcProcessor,
    'csl': CslProcessor,
    'wsc': WscProcessor,
    'copa': CopaProcessor,
}


def load_dataset_by_task_name(task_name, clue_dir):
    task_name = task_name.lower()
    data_dir = os.path.join(clue_dir, task_name)
    processor = CLUE_PROCESSORS[task_name]()
    type_name = processor.get_type()
    trains = processor.get_train_examples(data_dir)
    devs = processor.get_dev_examples(data_dir)
    tests = processor.get_test_examples(data_dir)
    return trains, devs, tests, type_name


