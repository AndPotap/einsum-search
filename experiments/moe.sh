# Uses 8 A100 GPUs
DATA_DIR=./open_small # TODO
OUT_DIR=./ # TODO
WANDB_PROJ=moe

BATCH_SIZE=64
GRAD_ACCUM=8
MAX_ITERS=100_000
BLOCK_SIZE=128

# BTT MoE
lr=3e-3
num_active_experts=2
struct="btt_norm_moe_para" # try low rank or dense MoE with "low_rank_moe" or "dense_moe"
for n_layer in 3 6 9; do
for d_model in 256 512 1024 2048; do
for num_experts in 8 16; do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$(shuf -i 49152-65535 -n 1) \
train_gpt.py config/train_open_small.py --block_size=${BLOCK_SIZE} --struct=${struct} --num_experts=${num_experts} --num_active_experts=${num_active_experts} --layers=all_but_last --d_model=${d_model} --n_layer=${n_layer} --n_head=-1 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${lr} --wandb_project=${WANDB_PROJ}
done;
done;
done;

# Standard MoE baseline
num_active_ffn_experts=2
for n_layer in 3 6 9; do
for d_model in 128 256 512 1024; do
for num_ffn_experts in 8 16; do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=$(shuf -i 49152-65535 -n 1) \
train_gpt.py config/train_open_small.py --block_size=${BLOCK_SIZE} --struct=dense --num_ffn_experts=${num_ffn_experts} --num_active_ffn_experts=${num_active_ffn_experts} --layers=all_but_last --d_model=${d_model} --n_layer=${n_layer} --n_head=-1 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${lr} --wandb_project=${WANDB_PROJ}
done;
done;
done;