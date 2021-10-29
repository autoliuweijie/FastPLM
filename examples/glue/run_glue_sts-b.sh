#!/bin/sh

# --task can be cola, sst-2, mrpc, sts-b, qqp, mnli, mnli-m, qnli, rte, wnli


export GLUE_DIR=../../../../datasets/academic/glue
export TASK_NAME=sts-b
export KERNEL_NAME=google_bert_base_en
CUDA_VISIBLE_DEVICES='0' nohup python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --pooling first --seq_length 256 \
    --batch_size 32 --learning_rate 2e-5 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK_NAME}_${KERNEL_NAME}.log &
echo "Launching ${KERNEL_NAME} with ${TASK_NAME}."


export GLUE_DIR=../../../../datasets/academic/glue
export TASK_NAME=sts-b
export KERNEL_NAME=google_albert_base_en
CUDA_VISIBLE_DEVICES='1' nohup python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --pooling first --seq_length 256 \
    --batch_size 48 --learning_rate 2e-5 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK_NAME}_${KERNEL_NAME}.log &
echo "Launching ${KERNEL_NAME} with ${TASK_NAME}."


export GLUE_DIR=../../../../datasets/academic/glue
export TASK_NAME=sts-b
export KERNEL_NAME=uer_gpt_en
CUDA_VISIBLE_DEVICES='2' nohup python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --pooling mean --seq_length 256 \
    --batch_size 32 --learning_rate 2e-5 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK_NAME}_${KERNEL_NAME}.log &
echo "Launching ${KERNEL_NAME} with ${TASK_NAME}."


export GLUE_DIR=../../../../datasets/academic/glue
export TASK_NAME=sts-b
export KERNEL_NAME=uer_gcnn_9_en
CUDA_VISIBLE_DEVICES='3' nohup python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --pooling mean --seq_length 256 \
    --batch_size 64 --learning_rate 2e-5 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train \
    > ./logs/${TASK_NAME}_${KERNEL_NAME}.log &
echo "Launching ${KERNEL_NAME} with ${TASK_NAME}."

