#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_trainval.py"
main_cfg_path="configs/megadepth_train.py"

#n_nodes=1
#n_gpus_per_node=1
n_gpus=4
torch_num_workers=16
batch_size=1
pin_memory=true
exp_name="outdoor-bs=$(($n_gpus * $batch_size))"

python -u ./train.py "configs/megadepth_train.py" \
    --exp_name="megadepth_bs4" \
    --accelerator="gpu" --devices=${n_gpus} --precision=32 \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=9000 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=40
