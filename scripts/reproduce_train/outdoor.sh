#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

main_cfg_path=$1 #"configs/megadepth_train.py"

n_gpus=4
torch_num_workers=4
batch_size=1
pin_memory=true
exp_name="outdoor-bs=$(($n_gpus * $batch_size))"

python -u ./train.py ${main_cfg_path} \
    --exp_name="megadepth_bs4" \
    --accelerator="gpu" --devices=${n_gpus} --precision=32 \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=9000 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=40
