# GLUE

Before running any one of these GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running the following lines at the root of the repo

```sh
python3 download_glue_data.py --data_dir /path/to/glue --tasks all
```

after replacing path/to/glue with a value that you like. Then you can run

```sh
export GLUE_DIR=/path/to/glue
export TASK_NAME=SST-2
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 32 --learning_rate 2e-5 \
    --finetuning_epochs_num 10 --distilling_epochs_num 5 \
    --load_pretrained_model --do_train
```

