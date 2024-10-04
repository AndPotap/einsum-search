### do "conda activate gpt" before running this script
DATA_DIR=/vast/ap6604/open_small
OUT_DIR=./
WANDB_PROJ=open_small

BATCH_SIZE=512
GRAD_ACCUM=1
MAX_ITERS=100_000
BLOCK_SIZE=128

# theta: (α, β, γ, δ, ε, φ, ρ)

### BTT ####
# full rank BTT
btt1="(0.5|0|0.5|0|0.5|0.5|0)"
btt2="(0.33|0|0.67|0|0.67|0.33|0)"
btt3="(0.25|0|0.75|0|0.75|0.25|0)"
btt5="(0.67|0|0.33|0|0.33|0.67|0)"
btt6="(0.75|0|0.25|0|0.25|0.75|0)"

# low rank BTT
btt7="(0.67|0|0.33|0|0.67|0.33|0)"
btt8="(0.33|0|0.67|0|0.33|0.67|0)"
btt9="(0.75|0|0.25|0|0.75|0.25|0)"
btt10="(0.25|0|0.75|0|0.25|0.75|0)"

# btt_rank > 1
btt11="(0.5|0|0.5|0|0.5|0.5|0.25)"
btt12="(0.5|0|0.5|0|0.5|0.5|0.33)"
btt13="(0.33|0|0.67|0|0.67|0.33|0.25)"
btt14="(0.67|0|0.33|0|0.33|0.67|0.25)"
btt15="(0.67|0|0.33|0|0.67|0.33|0.25)"
btt16="(0.33|0|0.67|0|0.33|0.67|0.25)"

#### TT/Kron ####
tt1="(0.5|0.5|0|0.5|0.5|0|0)"
tt2="(0.33|0.67|0|0.33|0.67|0|0)" 
tt3="(0.5|0.5|0|0.5|0.5|0|0.25)"
tt4="(0.33|0.67|0|0.33|0.67|0|0.25)" 

#### Low Rank ####
l1="(1|0|0|0|1|0|0.75)"
l2="(1|0|0|0|1|0|0.5)"

#### Generic ####
g1="(0.5|0.25|0.25|0.25|0.5|0.25|0)"
g2="(0.5|0.25|0.25|0.25|0.5|0.25|0.25)"

thetas="${btt1} ${btt2} ${btt3} ${btt5} ${btt6} ${btt7} ${btt8} ${btt9} ${btt10} ${btt11} ${btt12} ${btt13} ${btt14} ${btt15} ${btt16} ${tt1} ${tt2} ${tt3} ${tt4} ${l1} ${l2} ${g1} ${g2}"

# Einsums
lr=3e-3
for theta in ${thetas}; do
for n_layer in 3 6; do
for d_model in 256 512 768 1024 2048 4096; do
CUDA_VISIBLE_DEVICES=0 python train_gpt.py config/train_open_small.py --block_size=${BLOCK_SIZE} --struct=einsum_norm --expr=${theta} --layers=all_but_last --d_model=${d_model} --n_layer=${n_layer} --n_head=-1 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${lr} --wandb_project=${WANDB_PROJ}
done
done
done

# Dense
lr=3e-3
for theta in ${thetas}; do
for n_layer in 3 6; do
for d_model in 256 512 768 1024; do
CUDA_VISIBLE_DEVICES=0 python train_gpt.py config/train_open_small.py --block_size=${BLOCK_SIZE} --struct=dense --layers=all_but_last --d_model=${d_model} --n_layer=${n_layer} --n_head=-1 --d_head=64 --max_iters=${MAX_ITERS} --data_dir=${DATA_DIR} --out_dir=${OUT_DIR} --batch_size=${BATCH_SIZE} --gradient_accumulation_steps=${GRAD_ACCUM} --init_lr=${lr} --wandb_project=${WANDB_PROJ}
done
done
done
