#!/bin/sh

KERNEL='google_bert_base_zh'
TASK='tnews'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='0' python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling first --seq_length 256 \
    --batch_size 32 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train > ./logs/${TASK}_${KERNEL}.log &


KERNEL='google_albert_base_zh'
TASK='tnews'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='3' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling first --seq_length 256 \
    --batch_size 32 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train > ./logs/${TASK}_${KERNEL}.log &


KERNEL='uer_gpt_zh'
TASK='tnews'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='6' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling mean --seq_length 256 \
    --batch_size 32 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train > ./logs/${TASK}_${KERNEL}.log &


KERNEL='uer_gcnn_9_zh'
TASK='tnews'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='7' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling mean --seq_length 256 \
    --batch_size 32 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train > ./logs/${TASK}_${KERNEL}.log &
