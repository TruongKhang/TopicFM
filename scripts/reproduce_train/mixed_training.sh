#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

main_cfg_path=$1 #"configs/megadepth_train.py"
main_cfg_path2=$2

n_gpus=2
torch_num_workers=8
batch_size=1
pin_memory=true
exp_name="outdoor-bs=$(($n_gpus * $batch_size))"

python -u ./train.py --main_cfg_path=${main_cfg_path} --main_cfg_path2=${main_cfg_path2} \
    --exp_name="mixed_dataset" --n_gpus=${n_gpus} \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --max_epochs=40 --ckpt_path="logs/tb_logs/mixed_dataset/version_1/checkpoints/last.ckpt" # "logs/tb_logs/megadepth/version_11/checkpoints/last.ckpt"
