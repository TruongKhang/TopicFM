#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

#data_cfg_path="configs/data/megadepth_test_1500.py"
main_cfg_path="configs/megadepth_test.py"
ckpt_path="pretrained/model_ep29.ckpt"
dump_dir="dump/loftr_ds_outdoor"
profiler_name="inference"
n_nodes=1  # mannually keep this the same with --nodes
n_gpus_per_node=1
torch_num_workers=4
batch_size=1  # per gpu

python -u ./test.py \
    ${main_cfg_path} \
    --ckpt_path=${ckpt_path} \
    --dump_dir=${dump_dir} \
    --devices=1 --accelerator="gpu" --strategy="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers}\
    --profiler_name=${profiler_name} \
    --benchmark 
    
