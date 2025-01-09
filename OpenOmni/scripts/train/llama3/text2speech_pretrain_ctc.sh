#!/bin/bash
set -x

# wandb login
echo "DLC seemed world size(actually nodes): ${WORLD_SIZE}"
NNODES=${WORLD_SIZE}
NODE_RANK=${RANK}

export CUDA_DEVICE_MAX_CONNECTIONS=1
export GPUS_PER_NODE=8
# export NNODES=2
export BATCH_SIZE=1
export GRADIENT_ACCU_STEPS=4
export MASTER_PORT=29588
export CPUS_PER_TASK=16
export QUOTA=reserved

export DATA_PATH=/mnt/workspace/lr/datasets/openomni/json/openomni_stage3-1.json
export SAVE_PATH=openomni_stage3-1_qwen_2_ctc
export BASE_LR=2e-4
export VIT_LR=2e-4

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# 定义重试的最大次数
MAX_RETRIES=3

# 每次重试之间的等待时间，单位为秒
WAIT_TIME=20

# 当前的重试次数
retry_count=0

command_to_run="torchrun --nproc_per_node $GPUS_PER_NODE openomni/train/train_mem.py \
--deepspeed ./scripts/zero2.json \
--model_name_or_path /mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage3_qwen_3/checkpoint-20900 \
--version llava_qwen_2 \
--data_path ${DATA_PATH} \
--image_folder /mnt/workspace/lr/datasets \
--speech_folder /mnt/workspace/lr/datasets \
--vision_tower /mnt/workspace/lr/datasets/checkpoints/openai/clip-vit-large-patch14-336 \
--mm_projector_type mlp2x_gelu \
--freeze_backbone True \
--tune_mm_mlp_adapter False \
--freeze_mm_mlp_adapter True \
--tune_speech_generator_only True \
--pretrain_mm_mlp_adapter /mnt/workspace/lr/workspace/LLaVA_Her/checkpoints/openomni_stage2_qwen_2/mm_projector.bin \
--speech_encoder /mnt/workspace/lr/datasets/checkpoints/llava_her_pretrained/large-v3.pt \
--speech_generator /mnt/workspace/lr/datasets/checkpoints/iic/CosyVoice2-0.5B/llm.pt \
--unfreeze_mm_vision_tower False \
--mm_vision_tower_lr ${VIT_LR} \
--speech_generator_lr ${VIT_LR} \
--mm_projector_lr ${VIT_LR} \
--image_aspect_ratio anyres \
--group_by_modality_length True \
--mm_vision_select_layer -2 \
--mm_vision_select_feature patch \
--mm_patch_merge_type spatial_unpad \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir checkpoints/${SAVE_PATH} \
--num_train_epochs 2 \
--per_device_train_batch_size ${BATCH_SIZE} \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps ${GRADIENT_ACCU_STEPS} \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 1000 \
--save_total_limit 20 \
--learning_rate ${BASE_LR} \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 8196 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--run_name ${SAVE_PATH} \
--dataloader_drop_last True \
--speech_generator_type "ctc" \
--unit_vocab_size 6561 \
--report_to tensorboard | tee train_${SAVE_PATH}.log"

eval $command_to_run


