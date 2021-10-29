#!/bin/sh

:<<EOF
KERNEL='google_albert_base_zh'
TASK='csl'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='3' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling mean --seq_length 512 \
    --batch_size 24 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK}_${KERNEL}.log &
EOF


KERNEL='uer_gcnn_9_zh'
TASK='csl'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='4' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling mean --seq_length 512 \
    --batch_size 24 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK}_${KERNEL}.log &


:<<EOF
KERNEL='google_bert_base_zh'
TASK='csl'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling first --seq_length 512 \
    --batch_size 16 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK}_${KERNEL}.log &


KERNEL='uer_gpt_zh'
TASK='csl'
CLUE_PATH='../../../../datasets/academic/Chinese/clue/'
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_classifier.py --kernel $KERNEL --task $TASK --clue_dir $CLUE_PATH \
    --pooling mean --seq_length 512 \
    --batch_size 16 --learning_rate 1e-4 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK}_${KERNEL}.log &
EOF

