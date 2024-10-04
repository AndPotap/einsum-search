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


LOG_FREQ=1000
STEPS=1000000
WANDB_PROJ=synth

seed=0
input_dim=8
target_layers=6
target_width=1024
target_act=relu
lr=1e-3

### Einsums ####
depth=3
width=64
struct=einsum
layers=intermediate
for scale_factor in 1 2 8 16 32 64; do
for theta in ${thetas}; do
CUDA_VISIBLE_DEVICES=0 python3 train_synth.py \
--wandb_project=${WANDB_PROJ} \
--seed=${seed} \
--mixup=0 \
--smooth=0 \
--no-augment \
--dataset=${target_act} \
--input_dim=${input_dim} \
--target_layers=${target_layers} \
--target_width=${target_width} \
--target_act=${target_act} \
--model=MLP \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=4096 \
--calculate_stats=${LOG_FREQ} \
--steps=${STEPS} \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--expr=${theta} \
--layers=${layers} \
--scheduler=cosine
done;
done;