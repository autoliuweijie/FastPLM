#!/bin/sh

# --task can be cola, sst-2, mrpc, sts-b, qqp, mnli, mnli-m, qnli, rte, wnli

:<<EOF
export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=mnli
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

Evaluating model on mnli-dev task.
|mnli/acc=0.831890|speed=0.000000|ave_exec_layer=12.000000|cost_time=402.528842|
|mnli/acc=0.829343|speed=0.100000|ave_exec_layer=7.185227|cost_time=249.641264|
|mnli/acc=0.826999|speed=0.200000|ave_exec_layer=5.863780|cost_time=205.991362|
|mnli/acc=0.822618|speed=0.300000|ave_exec_layer=4.942944|cost_time=177.340946|
|mnli/acc=0.812532|speed=0.400000|ave_exec_layer=4.192970|cost_time=151.190741|
|mnli/acc=0.802241|speed=0.500000|ave_exec_layer=3.524605|cost_time=131.092809|
|mnli/acc=0.782170|speed=0.600000|ave_exec_layer=2.898828|cost_time=115.270808|
|mnli/acc=0.753133|speed=0.700000|ave_exec_layer=2.275802|cost_time=94.304070|
|mnli/acc=0.715945|speed=0.800000|ave_exec_layer=1.738054|cost_time=78.700901|
|mnli/acc=0.682425|speed=0.900000|ave_exec_layer=1.311156|cost_time=63.796805|
|mnli/acc=0.646154|speed=1.000000|ave_exec_layer=1.000000|cost_time=53.894047|
EOF


:<<EOF
export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=sst-2
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

|acc=0.922018|speed=0.000000|ave_exec_layer=12.000000|cost_time=33.869553|
|acc=0.917431|speed=0.100000|ave_exec_layer=3.444954|cost_time=11.025326|
|acc=0.902523|speed=0.200000|ave_exec_layer=2.628440|cost_time=8.860269|
|acc=0.894495|speed=0.300000|ave_exec_layer=2.240826|cost_time=7.925746|
|acc=0.885321|speed=0.400000|ave_exec_layer=1.941514|cost_time=7.122072|
|acc=0.880734|speed=0.500000|ave_exec_layer=1.646789|cost_time=6.340296|
|acc=0.868119|speed=0.600000|ave_exec_layer=1.483945|cost_time=5.885768|
|acc=0.857798|speed=0.700000|ave_exec_layer=1.349771|cost_time=5.527002|
|acc=0.855505|speed=0.800000|ave_exec_layer=1.217890|cost_time=5.176982|
|acc=0.854358|speed=0.900000|ave_exec_layer=1.128440|cost_time=4.934331|
|acc=0.847477|speed=1.000000|ave_exec_layer=1.000000|cost_time=4.600820|
EOF

:<<EOF
export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=mrpc
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

|acc=0.857843|f1=0.900344|acc_and_f1=0.879093|speed=0.000000|ave_exec_layer=12.000000|cost_time=16.126981|
|acc=0.845588|f1=0.892308|acc_and_f1=0.868948|speed=0.100000|ave_exec_layer=5.843137|cost_time=8.458705|
|acc=0.825980|f1=0.879865|acc_and_f1=0.852923|speed=0.200000|ave_exec_layer=3.960784|cost_time=6.207071|
|acc=0.786765|f1=0.854758|acc_and_f1=0.820761|speed=0.300000|ave_exec_layer=3.085784|cost_time=5.011639|
|acc=0.762255|f1=0.839669|acc_and_f1=0.800962|speed=0.400000|ave_exec_layer=2.531863|cost_time=4.237841|
|acc=0.745098|f1=0.828383|acc_and_f1=0.786740|speed=0.500000|ave_exec_layer=2.107843|cost_time=3.679638|
|acc=0.723039|f1=0.813839|acc_and_f1=0.768439|speed=0.600000|ave_exec_layer=1.899510|cost_time=3.441860|
|acc=0.718137|f1=0.811166|acc_and_f1=0.764652|speed=0.700000|ave_exec_layer=1.678922|cost_time=3.162433|
|acc=0.703431|f1=0.800000|acc_and_f1=0.751716|speed=0.800000|ave_exec_layer=1.507353|cost_time=2.943438|
|acc=0.678922|f1=0.782753|acc_and_f1=0.730837|speed=0.900000|ave_exec_layer=1.348039|cost_time=2.746694|
|acc=0.632353|f1=0.743151|acc_and_f1=0.687752|speed=1.000000|ave_exec_layer=1.000000|cost_time=2.316364|
EOF

:<<EOF
export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=sts-b
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

|pearson=0.851343|spearmanr=0.849518|corr=0.850430|speed=0.000000|ave_exec_layer=12.000000|cost_time=61.105391|
|pearson=0.851343|spearmanr=0.849518|corr=0.850430|speed=0.100000|ave_exec_layer=12.000000|cost_time=61.345298|
|pearson=0.851343|spearmanr=0.849518|corr=0.850430|speed=0.200000|ave_exec_layer=12.000000|cost_time=60.913657|
|pearson=0.851343|spearmanr=0.849518|corr=0.850430|speed=0.300000|ave_exec_layer=11.042667|cost_time=56.626991|
|pearson=0.852256|spearmanr=0.850514|corr=0.851385|speed=0.400000|ave_exec_layer=9.804000|cost_time=51.387319|
|pearson=0.841735|spearmanr=0.839946|corr=0.840840|speed=0.500000|ave_exec_layer=9.012000|cost_time=46.746814|
|pearson=0.830870|spearmanr=0.829822|corr=0.830346|speed=0.600000|ave_exec_layer=6.817333|cost_time=36.499590|
|pearson=0.799980|spearmanr=0.797672|corr=0.798826|speed=0.700000|ave_exec_layer=3.151333|cost_time=18.815104|
|pearson=0.691728|spearmanr=0.684292|corr=0.688010|speed=0.800000|ave_exec_layer=1.926667|cost_time=12.829595|
|pearson=0.505375|spearmanr=0.506400|corr=0.505888|speed=0.900000|ave_exec_layer=1.350667|cost_time=10.018835|
|pearson=0.253886|spearmanr=0.254247|corr=0.254067|speed=1.000000|ave_exec_layer=1.000000|cost_time=8.289719|
EOF

:<<EOF
export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=qqp
export KERNEL_NAME=google_bert_base_en

python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

load数据存在bug, 待赵欣修复
EOF

export GLUE_DIR=/root/workspace/datasets/academic/glue
export TASK_NAME=mnli-mm
export KERNEL_NAME=google_bert_base_en

:<<EOF
python3 -u run_glue.py \
    --kernel $KERNEL_NAME \
    --task $TASK_NAME \
    --glue_dir $GLUE_DIR \
    --batch_size 16 --learning_rate 2e-5 \
    --finetuning_epochs_num 5 --distilling_epochs_num 10 \
    --load_pretrained_model --do_train

|mnli-mm/acc=0.829638|speed=0.000000|ave_exec_layer=12.000000|cost_time=396.235510|
|mnli-mm/acc=0.828112|speed=0.100000|ave_exec_layer=7.923210|cost_time=263.499104|
|mnli-mm/acc=0.825061|speed=0.200000|ave_exec_layer=6.295464|cost_time=213.890126|
|mnli-mm/acc=0.823230|speed=0.300000|ave_exec_layer=5.219386|cost_time=189.116214|
|mnli-mm/acc=0.816416|speed=0.400000|ave_exec_layer=4.372966|cost_time=160.689384|
|mnli-mm/acc=0.807669|speed=0.500000|ave_exec_layer=3.636900|cost_time=143.833556|
|mnli-mm/acc=0.787632|speed=0.600000|ave_exec_layer=2.969284|cost_time=137.877343|
|mnli-mm/acc=0.765358|speed=0.700000|ave_exec_layer=2.325976|cost_time=112.847227|
|mnli-mm/acc=0.734845|speed=0.800000|ave_exec_layer=1.783971|cost_time=93.698413|
|mnli-mm/acc=0.691517|speed=0.900000|ave_exec_layer=1.307567|cost_time=75.008899|
|mnli-mm/acc=0.652055|speed=1.000000|ave_exec_layer=1.000000|cost_time=55.170583|
EOF

