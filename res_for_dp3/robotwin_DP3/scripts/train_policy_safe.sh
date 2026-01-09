#!/bin/bash

DEBUG=False
save_ckpt=True

alg_name=${1}
# task choices: See TASK.md
task_name=${2}
setting=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"

gpu_id=${7}
batch_size=${8:-32}  # Default batch size of 32 for memory safety

echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"
echo -e "\033[33mbatch size: ${batch_size}\033[0m"

if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode (Memory Safe)\033[0m"
fi

cd 3D-Diffusion-Policy

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Add memory optimization flags
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=4

python train_dp3.py --config-name=${config_name}.yaml \
                            task_name=${task_name} \
                            hydra.run.dir=${run_dir} \
                            training.debug=$DEBUG \
                            training.seed=${seed} \
                            training.device="cuda:0" \
                            exp_name=${exp_name} \
                            logging.mode=${wandb_mode} \
                            checkpoint.save_ckpt=${save_ckpt} \
                            expert_data_num=${expert_data_num} \
                            setting=${setting} \
                            dataloader.batch_size=${batch_size} \
                            val_dataloader.batch_size=${batch_size} \
                            dataloader.num_workers=4 \
                            val_dataloader.num_workers=4 \
                            training.gradient_accumulate_every=4