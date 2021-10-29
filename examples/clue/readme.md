# [CLUE](https://github.com/CLUEbenchmark/CLUE)

Before running any one of these GLUE tasks you should download the GLUE data by running the following lines at the root of the repo
```sh
python3 download_clue_data.py --data_dir /path/to/clue --tasks all
```

after replacing path/to/glue with a value that you like. Then you can run
```sh
export CLUE_DIR=/path/to/clue
export TASK_NAME=csl
export KERNEL_NAME=google_bert_base_zh

python3 -u run_classifier.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --clue_dir $GLUE_DIR \
    --batch_size 32 --learning_rate 1e-4 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

