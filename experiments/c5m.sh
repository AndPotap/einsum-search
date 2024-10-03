ds=cifar5m
lr=3e-3

### Dense ####
depth=3
width=64
struct=dense
for scale_factor in 0.5 1 2 4 8; do
python3 train_onepass.py \
--wandb_project=${ds} \
--seed=0 \
--dataset=${ds} \
--ar_modeling \
--mixup=0 \
--smooth=0 \
--no-augment \
--model=ARTransformer \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=64 \
--calculate_stats=800 \
--resolution=8 \
--epochs=2 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--scheduler=cosine
done;


### BTT full rank ###
btt1="(0.5|0|0.5|0|0.5|0.5|0)"
### BTT full rank > 1 btt-rank ###
btt2="(0.5|0|0.5|0|0.5|0.5|0.25)"
### BTT low rank ###
btt3="(0.67|0|0.33|0|0.67|0.33|0)"
#### Low Rank ####
l1="(1|0|0|0|1|0|0.5)"
#### TT ####
tt1="(0.5|0.5|0|0.5|0.5|0|0)"
tt2="(0.33|0.67|0|0.33|0.67|0|0)" 

thetas="${btt1} ${btt2} ${btt3} ${l1} ${tt1} ${tt2}"
struct=simple_ein_vec_norm
for theta in ${thetas}; do
for scale_factor in 0.5 1 2 4 8 16; do
python3 train_onepass.py \
--wandb_project=${ds} \
--seed=0 \
--dataset=${ds} \
--ar_modeling \
--mixup=0 \
--smooth=0 \
--no-augment \
--model=ARTransformer \
--width=${width} \
--depth=${depth} \
--lr=${lr} \
--batch_size=64 \
--calculate_stats=800 \
--resolution=8 \
--epochs=2 \
--optimizer=adamw \
--scale_factor=${scale_factor} \
--struct=${struct} \
--expr=${theta} \
--scheduler=cosine
done;
done;