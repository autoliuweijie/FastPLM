# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import time
from .config import *
from .utils import *
from .uer.utils.tokenizer import BertTokenizer
from .uer.utils.vocab import Vocab
from .uer.model_builder import build_model
from .uer.utils.optimizers import AdamW, WarmupLinearSchedule
from .uer.layers.multi_headed_attn import MultiHeadedAttention
from .uer.model_saver import save_model
from .uer.model_loader import load_model
from .fastbert import FastBERT, FastBERT_S2
from .classifiers import AttMiniClassifier


def clip_gradient(model,
                  clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


class FastGCNN(FastBERT):

    MiniClassifier = AttMiniClassifier

    def __init__(self,
                 kernel_name,
                 labels,
                 **kwargs):
        super(FastGCNN, self).__init__(kernel_name, labels, **kwargs)
        assert self.args.encoder == 'gatedcnn', 'encoder in args must be gatedcnn.'

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
        batch_size, seq_length, emb_size = embs_batch.size()
        masks_batch = self._mask_transfer(masks_batch, embs_batch)  # batch_size x seq_length x seq_length

        # gcnn encoder layer
        padding_batch = torch.zeros([batch_size, self.args.kernel_size-1, \
                                     emb_size]).to(embs_batch.device)
        embs_batch = torch.cat([padding_batch, embs_batch], dim=1).unsqueeze(1)  # batch_size, 1, seq_length+width-1, emb_size

        hidden_batch = self.kernel.encoder.conv_1(embs_batch)
        hidden_batch += self.kernel.encoder.conv_b1.repeat(1, 1, seq_length, 1)
        gate_batch = self.kernel.encoder.gate_1(embs_batch)
        gate_batch += self.kernel.encoder.gate_b1.repeat(1, 1, seq_length, 1)
        hidden_batch = hidden_batch * torch.sigmoid(gate_batch)
        res_input_batch = hidden_batch

        padding_batch = torch.zeros([batch_size, self.args.hidden_size, \
            self.args.kernel_size-1, 1]).to(embs_batch.device)
        hidden_batch = torch.cat([padding_batch, hidden_batch], dim=2)

        teacher_idx = self.kernel.encoder.layers_num - 2
        if labels_batch is not None:
            # training backbone of fastgcnn

            label_ids_batch = [self.label_map[label] for label in labels_batch]
            label_ids_batch = torch.tensor(label_ids_batch, dtype=torch.int64,
                    device=self.args.device)

            for i in range(self.kernel.encoder.layers_num - 1):
                hidden_tmp_batch = self.kernel.encoder.conv[i](hidden_batch)
                hidden_tmp_batch += self.kernel.encoder.conv_b[i].repeat(1, 1, seq_length, 1)
                gate_batch = self.kernel.encoder.gate[i](hidden_batch)
                gate_batch += self.kernel.encoder.gate_b[i].repeat(1, 1, seq_length, 1)
                hidden_batch = hidden_tmp_batch * torch.sigmoid(gate_batch)
                if (i+1) % self.args.block_size == 0:
                    hidden_batch = hidden_batch + res_input_batch
                    res_input_batch = hidden_batch
                hidden_batch = torch.cat([padding_batch, hidden_batch], dim=2)

            hidden_batch = hidden_batch[:,:,self.args.kernel_size-1:,:]
            output_batch = hidden_batch.transpose(1,2).contiguous().\
                view(batch_size, seq_length, self.args.hidden_size)
            logits_batch = self.classifiers[teacher_idx](output_batch, masks_batch)
            loss = self.criterion(
                    self.softmax(logits_batch.view(-1, self.labels_num)),
                    label_ids_batch.view(-1))

            return loss

        else:

            # distilating the student classifiers
            hidden_batch_list = []
            with torch.no_grad():
                for i in range(self.kernel.encoder.layers_num - 1):
                    hidden_tmp_batch = self.kernel.encoder.conv[i](hidden_batch)
                    hidden_tmp_batch += self.kernel.encoder.conv_b[i].repeat(1, 1, seq_length, 1)
                    gate_batch = self.kernel.encoder.gate[i](hidden_batch)
                    gate_batch += self.kernel.encoder.gate_b[i].repeat(1, 1, seq_length, 1)
                    hidden_batch = hidden_tmp_batch * torch.sigmoid(gate_batch)
                    if (i+1) % self.args.block_size == 0:
                        hidden_batch = hidden_batch + res_input_batch
                        res_input_batch = hidden_batch
                    hidden_batch = torch.cat([padding_batch, hidden_batch], dim=2)

                    output_batch = hidden_batch[:,:,self.args.kernel_size-1:,:]
                    output_batch = output_batch.transpose(1, 2).contiguous().\
                        view(batch_size, seq_length, self.args.hidden_size)
                    hidden_batch_list.append(output_batch)

                teacher_logits = self.classifiers[teacher_idx](
                        hidden_batch_list[teacher_idx], masks_batch
                    ).view(-1, self.labels_num)
                teacher_probs = F.softmax(teacher_logits, dim=1)

            loss = 0
            for i in range(self.kernel.encoder.layers_num - 2):
                student_logits = self.classifiers[i](
                        hidden_batch_list[i], masks_batch
                    ).view(-1, self.labels_num)
                loss += self.soft_criterion(
                        self.softmax(student_logits), teacher_probs)
            return loss

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
            batch_size, seq_length, emb_size = emb.size()
            mask = self._mask_transfer(mask, emb) # batch_size x seq_length x seq_length

            # gcnn encoder layer
            res_input = torch.transpose(emb.unsqueeze(3), 1, 2)
            padding = torch.zeros([batch_size, self.args.kernel_size-1, \
                                emb_size]).to(emb.device)
            emb = torch.cat([padding, emb], dim=1).unsqueeze(1)

            hidden = self.kernel.encoder.conv_1(emb)
            hidden += self.kernel.encoder.conv_b1.repeat(1, 1, seq_length, 1)
            gate = self.kernel.encoder.gate_1(emb)
            gate += self.kernel.encoder.gate_b1.repeat(1, 1, seq_length, 1)
            hidden = hidden * torch.sigmoid(gate)
            res_input = hidden

            padding = torch.zeros([batch_size, self.args.hidden_size, \
                    self.args.kernel_size-1, 1]).to(emb.device)
            hidden = torch.cat([padding, hidden], dim=2)

            teacher_idx = self.kernel.encoder.layers_num - 2 
            exec_layer_num = self.kernel.encoder.layers_num
            for i in range(self.kernel.encoder.layers_num - 1):
                hidden_tmp = self.kernel.encoder.conv[i](hidden)
                hidden_tmp += self.kernel.encoder.conv_b[i].repeat(1, 1, seq_length, 1)
                gate = self.kernel.encoder.gate[i](hidden)
                gate += self.kernel.encoder.gate_b[i].repeat(1, 1, seq_length, 1)
                hidden = hidden_tmp * torch.sigmoid(gate)
                if (i+1) % self.args.block_size == 0:
                    hidden = hidden + res_input
                    res_input = hidden
                hidden = torch.cat([padding, hidden], dim=2)

                output = hidden[:, :, self.args.kernel_size-1:, :]
                output = output.transpose(1, 2).contiguous().\
                        view(batch_size, seq_length, self.args.hidden_size)
                student_logits = self.classifiers[i](
                        output, mask).view(-1, self.labels_num)
                student_probs = F.softmax(student_logits, dim=1)
                uncertainty = calc_uncertainty(student_probs, labels_num=self.labels_num).item()
                
                if uncertainty < speed:
                    exec_layer_num = i + 2  # not i + 1
                    break

        label_id = torch.argmax(student_probs, dim=1).item()
        return label_id, exec_layer_num


class FastGCNN_S2(FastBERT_S2, FastGCNN):

    MiniClassifier = FastGCNN.MiniClassifier

    def _forward_for_loss(self,
                          sentences_batch,
                          labels_batch=None):
        return FastGCNN._forward_for_loss(self, sentences_batch, labels_batch)

    def _fast_infer(self,
                    sentence,
                    speed):
        return FastGCNN._fast_infer(self, sentence, speed)








