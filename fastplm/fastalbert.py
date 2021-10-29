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


class FastALBERT(FastBERT):

    MiniClassifier = DenseMiniClassifier
    def __init__(self,
                 kernel_name,
                 labels,
                 **kwargs):
        super(FastALBERT, self).__init__(kernel_name, labels, **kwargs)
        assert self.args.encoder == 'albert', 'encoder in args must be albert.'
        self.args.target == 'albert'

    def dev_step(self,
                 sentence): 
        ids, mask = self._convert_to_id_and_mask(sentence)
        self.eval()
        
        with torch.no_grad():
            ids = torch.tensor([ids], dtype=torch.int64, device=self.args.device)  # batch_size x seq_length
            mask = torch.tensor([mask], dtype=torch.int64, device=self.args.device)  # batch_size x seq_length
            # embedding layer
            emb = self.kernel.embedding(ids, mask)  # batch_size x seq_length x emb_size
            mask = self._mask_transfer(mask, emb) # batch_size x seq_length x seq_length
            # hidden layers
            speed=0.0
            hidden = self.kernel.encoder.linear(emb)
            exec_layer_num = self.kernel.encoder.layers_num
            for i in range(self.kernel.encoder.layers_num):
                hidden = self.kernel.encoder.transformer(hidden, mask) # batch_size x seq_length x seq_length
                logits = self.classifiers[i](hidden, mask)  # batch_size x labels_num
                probs = F.softmax(logits, dim=1) # batch_size x labels_num
                uncertainty = calc_uncertainty(probs, labels_num=self.labels_num,\
                        N=self.args.uncertainty_N).item()

                if uncertainty < speed:
                    exec_layer_num = i + 1
                    break

    def _fast_infer(self,
                    sentence,
                    speed):
        ids, mask = self._convert_to_id_and_mask(sentence)

        self.eval()
        with torch.no_grad():
            ids = torch.tensor([ids], dtype=torch.int64, device=self.args.device)  # batch_size x seq_length
            mask = torch.tensor([mask], dtype=torch.int64, device=self.args.device)  # batch_size x seq_length

            # embedding layer
            emb = self.kernel.embedding(ids, mask)  # batch_size x seq_length x emb_size
            mask = self._mask_transfer(mask, emb) # batch_size x seq_length x seq_length

            # hidden layers
            hidden = self.kernel.encoder.linear(emb)
            exec_layer_num = self.kernel.encoder.layers_num
            for i in range(self.kernel.encoder.layers_num):
                hidden = self.kernel.encoder.transformer(hidden, mask) # batch_size x seq_length x seq_length
                logits = self.classifiers[i](hidden, mask)  # batch_size x labels_num
                probs = F.softmax(logits, dim=1) # batch_size x labels_num
                uncertainty = calc_uncertainty(probs, labels_num=self.labels_num,\
                        N=self.args.uncertainty_N).item()

                if uncertainty < speed:
                    exec_layer_num = i + 1
                    break

        label_id = torch.argmax(probs, dim=1).item()
        return label_id, exec_layer_num
    
    def _forward_for_loss(self,
                          sentences_batch,
                          labels_batch=None):

        self.train()
        ids_batch, masks_batch = [], []
        for sentence in sentences_batch:
            ids, masks = self._convert_to_id_and_mask(sentence)
            ids_batch.append(ids)
            masks_batch.append(masks)
        ids_batch = torch.tensor(ids_batch, dtype=torch.int64, device=self.args.device)  # batch_size x seq_length
        masks_batch = torch.tensor(masks_batch, dtype=torch.int64, device=self.args.device)  # batch_size x seq_length

        # embedding layer
        embs_batch = self.kernel.embedding(ids_batch, masks_batch)  # batch_size x seq_length x emb_size
        masks_batch = self._mask_transfer(masks_batch, embs_batch)  # batch_size x seq_length x seq_length

        if labels_batch is not None:

            # training backbone of fastbert
            label_ids_batch = [self.label_map[label] for label in labels_batch]
            label_ids_batch = torch.tensor(label_ids_batch, dtype=torch.int64,
                    device=self.args.device)

            hiddens_batch = self.kernel.encoder.linear(embs_batch)
            for i in range(self.kernel.encoder.layers_num):
                hiddens_batch = self.kernel.encoder.transformer(
                        hiddens_batch, masks_batch)
            logits_batch = self.classifiers[-1](hiddens_batch, masks_batch)
            loss = self.criterion(
                    self.softmax(logits_batch.view(-1, self.labels_num)),
                    label_ids_batch.view(-1)
                )

            return loss

        else:

            # distilating the student classifiers
            hiddens_batch = self.kernel.encoder.linear(embs_batch)
            hiddens_batch_list = []
            with torch.no_grad():
                for i in range(self.kernel.encoder.layers_num):
                    hiddens_batch = self.kernel.encoder.transformer(
                            hiddens_batch, masks_batch)
                    hiddens_batch_list.append(hiddens_batch)
                teacher_logits = self.classifiers[-1](
                        hiddens_batch_list[-1], masks_batch
                    ).view(-1, self.labels_num)
                teacher_probs = F.softmax(teacher_logits, dim=1)

            loss = 0
            for i in range(self.kernel.encoder.layers_num - 1):
                student_logits = self.classifiers[i](
                        hiddens_batch_list[i], masks_batch
                    ).view(-1, self.labels_num)
                loss += self.soft_criterion(
                        self.softmax(student_logits), teacher_probs)
            return loss


class FastALBERT_S2(FastBERT_S2, FastALBERT):

    MiniClassifier = FastALBERT.MiniClassifier

    def _forward_for_loss(self,
                          sentences_batch,
                          labels_batch=None):
        return FastALBERT._forward_for_loss(self, sentences_batch, labels_batch)

    def _fast_infer(self,
                    sentence,
                    speed):
        return FastALBERT._fast_infer(self, sentence, speed)


