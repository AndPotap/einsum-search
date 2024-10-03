import os
import json
import wandb
import torch
import torch.nn.functional as F
from torch.nn import MSELoss
from model import set_seed
from nn.cola_nn import cola_parameterize, get_model_summary_and_flops
import nn
from tqdm import tqdm
from math import sqrt
from scaling_mlps.utils.config import config_to_name
from scaling_mlps.utils.parsers import get_training_parser
from scaling_mlps.utils.optimizer import get_scheduler

def my_relu(x):
    return torch.relu(x) * sqrt(2)
def my_sine(x):
    return torch.sin(x) * sqrt(2)

def create_target_function(args, device):
    set_seed(42)
    hidden_dim, input_dim, num_layers = args.target_width, args.input_dim, args.target_layers
    activation = my_relu if args.target_act == 'relu' else my_sine
    weights = [torch.randn(hidden_dim, input_dim if i == 0 else hidden_dim, device='cpu') * (args.hidden_w_std or (input_dim if i == 0 else hidden_dim)) ** -0.5 for i in range(num_layers)]
    biases = [None if args.hidden_b_std is None else torch.randn(hidden_dim, device='cpu') * args.hidden_b_std for _ in range(num_layers)]
    output_w = torch.randn(1, hidden_dim, device='cpu') * (hidden_dim ** -0.5)
    output_b = torch.zeros(1, device='cpu')

    def fn(X, weights, biases, output_w, output_b):
        h = X
        for w, b in zip(weights, biases):
            h = activation(F.linear(h, w, b))
        return F.linear(h, output_w, output_b)

    # run a forward pass to print mean and var of each layer
    Z = torch.randn(1024, input_dim, device='cpu')
    out = fn(Z, weights, biases, output_w, output_b)
    mean = out.mean()
    std = out.std() + 1e-8
    # scale last layer weights and add bias
    output_w = output_w / std
    output_b = -mean / std
    # print mean and var of each layer
    for w, b in zip(weights, biases):
        Z = activation(F.linear(Z, w, b))
        print(f'µ: {Z.mean().item():.1f}, σ: {Z.std().item():.1f}')
    out = F.linear(Z, output_w, output_b)
    print(f'µ: {out.mean().item():.1f}, σ: {out.std().item():.1f}')
    # move weights and biases to device
    weights = [w.to(device) for w in weights]
    biases = [b.to(device) if b is not None else None for b in biases]
    output_w = output_w.to(device)
    output_b = output_b.to(device)
    @torch.no_grad()
    def target_function(X):
        return fn(X, weights, biases, output_w, output_b)

    return target_function

def train_step(model, opt, scheduler, loss_fn, X, y, args):
    model.train()
    loss = loss_fn(model(X), y)
    loss.backward()
    if args.clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    opt.step()
    opt.zero_grad()
    scheduler.step()
    return loss.item()

@torch.no_grad()
def test_step(model, X, y, loss_fn):
    model.eval()
    return loss_fn(model(X), y).item()

def main(args):
    set_seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_shape = (1, args.input_dim)
    model_builder = getattr(nn, args.model)
    base_config = dict(dim_in=args.input_dim, dim_out=1, depth=args.depth, width=args.width, ffn_expansion=4, use_bias=args.use_bias)
    target_config = base_config.copy()
    target_config['width'] = int(args.width * args.scale_factor)
    args.width = int(args.width * args.scale_factor)  # we update args.width to have it logged in wandb
    target_config['ffn_expansion'] = args.ffn_expansion

    cola_kwargs = dict(tt_dim=args.tt_dim, tt_rank=args.tt_rank, num_blocks=args.num_blocks, rank_frac=args.rank_frac, expr=args.expr, init_type=args.init_type, do_sgd_lr=(args.optimizer=="sgd"))
    model, opt = cola_parameterize(model_builder, base_config, args.lr, target_config=target_config, struct=args.struct, layer_select_fn=args.layers, device=device, cola_kwargs=cola_kwargs, optim_kwargs={'weight_decay': args.weight_decay}, use_wrong_mult=args.use_wrong_mult)
    info = get_model_summary_and_flops(model, torch.zeros(*input_shape).to(device))

    run_name = config_to_name(args)
    path = os.path.join(args.checkpoint_folder, run_name)
    os.makedirs(path, exist_ok=True)
    with open(path + '/config.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    scheduler = get_scheduler(opt, args.scheduler, epochs=args.steps)
    loss_fn = MSELoss()
    target_function = create_target_function(args, device)
    compute_per_step = info['cola_flops'] * args.batch_size

    if args.wandb:
        config = args.__dict__
        config.update(info)
        wandb.init(project=args.wandb_project, name=run_name, config=config, tags=["regression"])

    recent_train_losses, recent_dh_norms = [], []
    prev_hs = None
    X_test = torch.randn(args.batch_size, args.input_dim, device=device)
    y_test = target_function(X_test)

    for step in (pb := tqdm(range(args.steps))):
        calc_stats = step % args.calculate_stats == 0
        X = torch.randn(args.batch_size, args.input_dim, device=device)
        y = target_function(X)
        train_loss = train_step(model, opt, scheduler, loss_fn, X, y, args)
        recent_train_losses.append(train_loss)

        if calc_stats:
            model.clear_features()
            test_loss = test_step(model, X_test, y_test, loss_fn)
            hs = [torch.cat(h.buffer, dim=0) for h in model.get_features()]
            if prev_hs is None:
                prev_hs = hs
            dhs = [hs[i] - prev_hs[i] for i in range(len(hs))]
            h_norm = [torch.norm(h, dim=1).mean() / h.shape[1]**0.5 for h in hs]
            dh_norm = [torch.norm(dh, dim=1).mean() / dh.shape[1]**0.5 for dh in dhs] if dhs else None
            recent_dh_norms.append(dh_norm)
            prev_hs = hs

            logs = {
                "step": step,
                "test_loss": test_loss,
                "train_loss": train_loss,
                "train_loss_avg": sum(recent_train_losses) / len(recent_train_losses),
                "current_compute": compute_per_step * step,
            }
            if args.wandb:
                logs.update({f'h_{i}': h_norm[i].item() for i in range(len(h_norm))})
                logs.update({f'dh_{i}': dh_norm[i].item() for i in range(len(dh_norm))})
                logs.update({f'dh_avg_{i}': torch.tensor(recent_dh_norms)[:, i].mean().item() for i in range(len(dh_norm))})
                logs.update({f'rms/{name}': p.rms for name, p in model.named_parameters() if hasattr(p, 'rms')})
                wandb.log(logs)

            pb.set_description(f"Step {step}, Train Loss: {train_loss:.1f}, Test Loss: {test_loss:.1f}")
            recent_train_losses = []

    if args.save:
        torch.save(model.state_dict(), path + "/final_checkpoint.pt")

if __name__ == "__main__":
    parser = get_training_parser()
    args = parser.parse_args()
    main(args) 