import numpy as np

dataset = 'open'
wandb_project = "open_small"
wandb_log = True

struct = "btt"
compile = False

init_lr = 3e-3
weight_decay = 0.
opt_name = "AdamW"

log_interval = 100
every_n_fwds = 200
data_dir = f"/datasets/{dataset}"
out_dir = f"out-open-{struct}"

batch_size = 12
block_size = 128
gradient_accumulation_steps = 5 * 8  # must be multiple of the world size

do_qk_ln = True
split_qkv = True
n_layer, n_head, d_model = 6, 12, 768
total_tokens = 6_017_506_009
eval_interval = 2_500

decay_lr = False
always_save_checkpoint = False
max_iters = 10_000
warmup_iters = 2_000

vocab_size = 96
data_dtype = np.uint8
head_btt_case_qkv = ""
head_btt_case_proj = ""
use_head_btt = False