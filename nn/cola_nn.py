from functools import reduce, partial
from collections import OrderedDict
import opt_einsum
import inspect
from math import prod
from math import ceil
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD
from fvcore.nn import FlopCountAnalysis
from torchinfo import summary
from sympy import factorint
from cola.ops import Dense
from ops.operators import OptBlockTT
from ops.operators import EinOpVec2
from learning.fns import gen_cores
from learning.fns import get_einsum_expr, get_einsum_exprs
from learning.fns import construct_vec_from_exps
from learning.fns import get_core_bmm_dims
try:
    from trainkit.saving import save_object
except ImportError:

    def save_object(*args, **kwargs):
        pass


try:
    from torchdistx.deferred_init import deferred_init, materialize_module
except ImportError:

    def deferred_init(builder, **kwargs):
        return builder(**kwargs)

    def materialize_module(*args, **kwargs):
        return None

    print("Unable to lazy init models")


def is_cola_param(x):
    if isinstance(x, torch.Tensor) or isinstance(x, nn.Parameter):
        return x.dtype == torch.float32 or x.dtype == torch.float64 or x.dtype == torch.float16
    return False


def dense_init(linear, zero_init=False):
    d_in, _ = linear.in_features, linear.out_features
    if zero_init:
        std = 0
        linear.weight.zero_init = True
    else:
        std = d_in**-0.5
    linear.weight.data.normal_(mean=0, std=std)
    if linear.bias is not None:
        linear.bias.data.zero_()


def cola_init(tensor, zero_init=False):
    assert hasattr(tensor, 'd_in'), 'Learnable CoLA parameter must have d_in attribute'
    if zero_init:
        print(f'Zero init cola param of shape {list(tensor.shape)}')
        with torch.no_grad():
            return tensor.zero_()
    else:
        if hasattr(tensor, 'already_init'):
            return tensor
        elif hasattr(tensor, 'init_std'):
            std = tensor.init_std
            print(f'Init cola param of shape {list(tensor.shape)} with std {std}')
        else:
            # if hasattr(tensor, 'd_out'):
            #     # sqrt(d_out / d_in) spectral norm
            #     std = np.sqrt(min(tensor.d_in, tensor.d_out) * tensor.d_out / (tensor.d_out * tensor.d_in * tensor.d_in))
            # else:
            std = np.sqrt(1 / tensor.d_in)
        with torch.no_grad():
            return tensor.normal_(0, std)


def factorize(x, n):
    # Get prime factors and their counts
    prime_factors = factorint(x)

    # Initialize the n integers
    numbers = [1] * n

    # Distribute the prime factors
    for prime, count in prime_factors.items():
        for _ in range(count):
            # Find the number with the smallest product to assign the prime factor
            min_index = min(range(n), key=lambda i: numbers[i])
            numbers[min_index] *= prime

    # return in ascending order
    return sorted(numbers)


def get_builder_fn(struct, **kwargs):
    build_fn = partial(build_fns[struct], **kwargs)
    return build_fn


class CoLALayer(nn.Module):
    def __init__(self, A, in_features=None, out_features=None, padding=False, bias=True, track_spectral_norm=False,
                 padding_tol=0.15, do_rms_norm=False):
        super().__init__()
        d_in_orig, d_out_orig = A.shape
        in_features = in_features or d_in_orig
        out_features = out_features or d_out_orig
        if d_in_orig != in_features:
            assert padding, f'Input dimension mismatch: {d_in_orig} != {in_features}, set padding=True'
            rel_pad = abs(d_in_orig - in_features) / in_features
            print(f'Relative padding {rel_pad:.2f}')
            assert rel_pad <= padding_tol, f'Relative padding {rel_pad:.2f} exceeds tolerance {padding_tol}'
        if d_out_orig != out_features:
            assert padding, f'Output dimension mismatch: {d_out_orig} != {out_features}, set padding=True'
            rel_pad = abs(d_out_orig - out_features) / out_features
            print(f'Relative padding {rel_pad:.2f}')
            assert rel_pad <= padding_tol, f'Relative padding {rel_pad:.2f} exceeds tolerance {padding_tol}'
        self.padding = padding
        self.do_rms_norm = do_rms_norm
        self.in_features = in_features
        self.out_features = out_features
        cola_tensors, self.unflatten = A.flatten()
        # learnable matrices parametrizing A
        self.matrix_params = nn.ParameterList()
        self.info = {}
        self.num_mats = sum(is_cola_param(t) and t.dim() > 0 for t in cola_tensors)  # ignore scalars
        # Iterate over cola_tensors and only turn those that meet the condition into Parameters
        # e.g. permutations are excluded
        self.cola_tensors = []
        for t in cola_tensors:
            if is_cola_param(t):
                if t.dim() > 0:
                    assert hasattr(t, 'd_in'), f'Learnable CoLA parameter {t} must have d_in attribute'
                    d_in = t.d_in
                    d_out = t.d_out
                    t = nn.Parameter(t)
                    t.d_in = d_in
                    t.d_out = d_out
                else:
                    t = nn.Parameter(t)
                self.cola_tensors.append(t)
                self.matrix_params.append(t)
            else:
                self.cola_tensors.append(t)
        self.A = self.unflatten(self.cola_tensors)  # (d_in, d_out)
        self.b = nn.Parameter(torch.zeros(d_out_orig)) if bias else None
        if bias:
            self.b.lr_mult = 1
        self.track_spectral_norm = track_spectral_norm
        if track_spectral_norm:
            # only works if "self.natural_norm" has non-zero gradient
            self.top_singular_vec = nn.Parameter(torch.randn(d_in_orig))
            self.top_singular_vec.lr_mult = 0
        self.info = {}

    def _apply(self, fn):
        # _apply is called when doing model.to(device), model.cuda(), model.float(), etc.
        # apply fn to all parameters
        super()._apply(fn)  # TODO: check what happens to non-parameter tensors? (e.g. do they get moved to device?)
        # reconstruct A
        self.A = self.unflatten(self.cola_tensors)
        return self

    def update_info(self, x, out):
        with torch.no_grad():
            p = self.matrix_params[0]
            p.x = torch.sqrt(torch.mean(x**2) + 1e-8).item()
            p.out = torch.sqrt(torch.mean(out**2) + 1e-8).item()
            p.scale = p.out / p.x
            for p in self.matrix_params:
                if p.dim() > 0:
                    p.rms = normalized_rms(p)
                else:
                    p.rms = p.abs().item()

    def forward(self, x):
        batch_shape = x.shape[:-1]
        x = x.view(-1, x.shape[-1])  # (B, d)
        if self.padding:
            x_dim, A_in = x.shape[-1], self.A.shape[0]
            if A_in > x_dim:
                x = F.pad(x, pad=(0, self.A.shape[0] - x.shape[-1]))
            else:
                x = x[:, :A_in]
        if self.track_spectral_norm:
            v = rms_norm(self.top_singular_vec[None, :], dim=-1)
            xv = torch.cat([x, v])  # (B + 1, d_in_orig)
            out = xv @ self.A  # (B + 1, d_out_orig)
            out, u = out[:-1], out[-1]  # (B, d_out_orig), (d_out_orig)
            self.natural_norm = rms(u, dim=-1)
        else:
            out = x @ self.A
        if self.b is not None:
            out = out + self.b
        if not self.training:
            self.update_info(x, out)
        if self.padding:
            out_dim = out.shape[-1]
            if out_dim >= self.out_features:
                out = out[:, :self.out_features]
            else:
                out = F.pad(out, pad=(0, self.out_features - out_dim))
        if self.do_rms_norm:
            out_rms = rms(out, dim=-1, eps=1e-6)  # (B,)
            denom = torch.max(out_rms, torch.ones_like(out_rms))[:, None]
            out = out / denom
        return out.view(*batch_shape, -1)


class CoLAMoELayer(nn.Module):
    def __init__(self, experts, k=2):
        super().__init__()
        assert len(experts) > 0
        assert all(isinstance(expert, CoLALayer) for expert in experts), "All experts must be CoLALayers"
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.num_active_experts = min(k, len(experts))
        self.in_features = experts[0].in_features
        self.out_features = experts[0].out_features
        self.moe_gate = nn.Linear(self.in_features, len(experts), bias=False)
        # zero init gate to make initial selection uniform and obey µP
        self.moe_gate.weight.data.zero_()
        # list of parameters that has structure-aware LR mult
        self.moe_gate.weight.lr_mult = 1
        self.matrix_params = [self.moe_gate.weight]
        for expert in self.experts:
            self.matrix_params.extend(expert.matrix_params)
        # assert num_mats identical for all experts
        assert all(expert.num_mats == self.experts[0].num_mats for expert in self.experts), "Experts must have same num_mats"
        self.num_mats = experts[0].num_mats

    def load_balancing_loss_fn(self, gate_logits, top_k_weights, selected_experts):
        probs = F.softmax(gate_logits, dim=1)  # (B, E)
        zeros = torch.zeros_like(probs)  # (B, E)
        zeros = zeros.to(top_k_weights.dtype)
        gates = zeros.scatter(1, selected_experts, top_k_weights)  # (B, E)
        return self.compute_aux_loss(probs, gate_logits, gates)

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.size(0)
        probs = probs.sum(0)  # unnormalized marginal expert probs
        freq = (gates > 0).float().sum(0)  # unnoramlized marginal freqs
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1))**2).sum()  # squared log partition functions
        switchloss = self.num_experts * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss
        return loss

    def forward(self, inputs):
        batch_shape = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.moe_gate(inputs)
        top_k_weights, selected_experts = torch.topk(gate_logits, self.num_active_experts, dim=-1)  # (B, k), (B, k)
        top_k_weights = F.softmax(top_k_weights, dim=1, dtype=torch.float).to(inputs.dtype)  # (B, k)
        self.moe_gate.load_balancing_loss = self.load_balancing_loss_fn(gate_logits, top_k_weights, selected_experts)
        if not self.training:
            # compute perplexity of gate logits
            gate_probs = F.softmax(gate_logits, dim=1)
            agg_gate_probs = gate_probs.mean(dim=0)
            ppl = torch.exp(-torch.sum(gate_probs * torch.log(gate_probs), dim=1))
            agg_ppl = torch.exp(-torch.sum(agg_gate_probs * torch.log(agg_gate_probs), dim=0))
            self.moe_gate.weight.ppl = ppl.mean().item()  # should be low
            self.moe_gate.weight.agg_ppl = agg_ppl.item()  # should be high

        # Compute indices and inputs for each expert
        active_expert_indices = torch.unique(selected_experts)
        active_experts = [self.experts[i] for i in active_expert_indices]
        batch_idxs = [torch.where(selected_experts == i)[0] for i in active_expert_indices]
        nth_experts = [torch.where(selected_experts == i)[1] for i in active_expert_indices]
        # expert_inputs = [inputs[batch_idx] for batch_idx in batch_idxs]

        expert_outputs = []
        # for expert_in, expert in zip(expert_inputs, active_experts):
        for batch_idx, expert in zip(batch_idxs, active_experts):
            expert_in = inputs[batch_idx]
            expert_outputs.append(expert(expert_in))

        results = torch.zeros(inputs.shape[0], self.out_features, device=inputs.device)
        for expert_output, batch_idx, nth_expert in zip(expert_outputs, batch_idxs, nth_experts):
            results[batch_idx] += top_k_weights[batch_idx, nth_expert, None] * expert_output
        return results.view(*batch_shape, -1)


class BTTMoELayer(nn.Module):
    def __init__(self, btt_layer, k=2):
        super().__init__()
        self.btt_layer = btt_layer
        self.A, self.b = btt_layer.A, btt_layer.b
        assert isinstance(self.A, OptBlockTT), "The operator must be an OptBlockTT"
        self.num_experts = self.A.rs[1]  # btt rank
        self.num_active_experts = min(k, self.num_experts)
        self.in_features = self.btt_layer.in_features
        self.out_features = self.btt_layer.out_features
        self.moe_gate = nn.Linear(self.in_features, self.num_experts, bias=False)
        # zero init gate to make initial selection uniform and obey µP
        self.moe_gate.weight.data.zero_()
        # list of parameters that has structure-aware LR mult
        self.moe_gate.weight.lr_mult = 1
        self.matrix_params = [self.moe_gate.weight]
        self.matrix_params.extend(self.btt_layer.matrix_params)
        self.num_mats = self.btt_layer.num_mats

    def load_balancing_loss_fn(self, gate_logits, sparse_weights):
        probs = F.softmax(gate_logits, dim=1)  # (B, E)
        return self.compute_aux_loss(probs, gate_logits, sparse_weights)

    def compute_aux_loss(self, probs, logits, sparse_weights):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.size(0)
        probs = probs.sum(0)  # unnormalized marginal expert probs
        freq = (sparse_weights > 0).float().sum(0)  # unnoramlized marginal freqs
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1))**2).sum()  # squared log partition functions

        switchloss = self.num_experts * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss
        return loss

    def forward(self, inputs):
        batch_shape = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.moe_gate(inputs)
        top_k_weights, selected_experts = torch.topk(gate_logits, self.num_active_experts, dim=-1)  # (B, k), (B, k)
        top_k_weights = F.softmax(top_k_weights, dim=1, dtype=torch.float).to(inputs.dtype)  # (B, k)
        top_k_weights = top_k_weights + 1e-6  # so they can be inverted safetly during backward
        if not self.training:
            # compute perplexity of gate logits
            gate_probs = F.softmax(gate_logits, dim=1)
            agg_gate_probs = gate_probs.mean(dim=0)
            ppl = torch.exp(-torch.sum(gate_probs * torch.log(gate_probs), dim=1))
            agg_ppl = torch.exp(-torch.sum(agg_gate_probs * torch.log(agg_gate_probs), dim=0))
            self.moe_gate.weight.ppl = ppl.mean().item()  # should be low
            self.moe_gate.weight.agg_ppl = agg_ppl.item()  # should be high
        sparse_weights = torch.zeros_like(gate_logits).to(inputs.dtype).scatter_(1, selected_experts, top_k_weights)  # (B, E)
        self.moe_gate.load_balancing_loss = self.load_balancing_loss_fn(gate_logits, sparse_weights)
        out = self.A._rmatmat(inputs, sparse_weights, self.num_active_experts)
        if self.b is not None:
            out = out + self.b
        if not self.training:
            self.update_info(inputs, out)
        return out.view(*batch_shape, -1)

    def update_info(self, x, out):
        with torch.no_grad():
            p = self.matrix_params[0]
            p.x = torch.sqrt(torch.mean(x**2) + 1e-8).item()
            p.out = torch.sqrt(torch.mean(out**2) + 1e-8).item()
            p.scale = p.out / p.x

def build_dense(d_in, d_out, bias=True, zero_init=False, **_):
    U = torch.randn(d_in, d_out)
    U.d_in = d_in
    U.d_out = d_out
    U.init_std = np.sqrt(min(U.d_in, U.d_out) * U.d_out / (U.d_out * U.d_in * U.d_in))
    cola_init(U, zero_init)
    return CoLALayer(Dense(U), bias=bias)

def build_low_rank(d_in, d_out, rank_frac=0, bias=True, zero_init=False, spect_init=False, **_):
    assert rank_frac >= 0, 'rank_frac must be non-negative'
    if rank_frac == 0:
        rank_frac = 1 / np.sqrt(min(d_in, d_out))
    rank = ceil(rank_frac * min(d_in, d_out))
    U = torch.randn(d_in, rank)
    U.d_in = d_in
    U.d_out = rank
    if spect_init:
        U.init_std = np.sqrt(min(U.d_in, U.d_out) * U.d_out / (U.d_out * U.d_in * U.d_in))
    V = torch.randn(rank, d_out)
    V.d_in = rank
    V.d_out = d_out
    cola_init(U, zero_init)
    cola_init(V)
    A = Dense(U) @ Dense(V)
    return CoLALayer(A, bias=bias)

def build_low_rank_norm(d_in, d_out, rank_frac=0, bias=True, zero_init=False, spect_init=False, **_):
    assert rank_frac >= 0, 'rank_frac must be non-negative'
    if rank_frac == 0:
        rank_frac = 1 / np.sqrt(min(d_in, d_out))
    rank = ceil(rank_frac * min(d_in, d_out))
    U = torch.randn(d_in, rank)
    U.d_in = d_in
    U.d_out = rank
    if spect_init:
        U.init_std = np.sqrt(min(U.d_in, U.d_out) * U.d_out / (U.d_out * U.d_in * U.d_in))
    V = torch.randn(rank, d_out)
    V.d_in = rank
    V.d_out = d_out
    cola_init(U, zero_init)
    cola_init(V)
    A = DenseNorm(U) @ DenseNorm(V)
    return CoLALayer(A, bias=bias)

def build_opt_btt(d_in, d_out, tt_dim=2, tt_rank=1, bias=True, permute=False, zero_init=False, normalize=False, learn_gamma=True,
                  init='smart_spect', **_):
    assert tt_rank > 0, 'tt_rank must be positive'
    ns, ms = tuple(factorize(d_out, tt_dim)), tuple(factorize(d_in, tt_dim))
    rs = (1, tt_rank, 1)
    shapes = (rs, ms, ns)
    cores = []
    print(f'TT shape: {ms} -> {ns}')
    for idx in range(tt_dim):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        core.d_out = ns[idx] * rs[idx + 1]
        if init == 'spect':
            core.init_std = np.sqrt(min(core.d_in, core.d_out) * core.d_out / (core.d_out * core.d_in * core.d_in))
        elif init == 'smart_spect':
            batch_dims = core.shape[:-2]
            data = core.data
            data = data.view(*batch_dims, rs[idx], ms[idx], rs[idx + 1] * ns[idx])
            d_in = ms[idx]
            d_out = rs[idx + 1] * ns[idx]
            std = np.sqrt(min(d_in, d_out) * d_out / (d_out * d_in * d_in))
            data = torch.randn_like(data) * std  # (..., r, m, r' * n)
            # zero out r > 0 elements
            data[..., 1:, :, :] = 0
            core.data = data.view(*core.shape)
            core.already_init = True
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)

    if normalize and learn_gamma:
        gamma0 = nn.Parameter(torch.tensor(1.))
        gamma1 = nn.Parameter(torch.tensor(1.))
        cores.append(gamma0)
        cores.append(gamma1)
    A = OptBlockTT(cores, shapes, normalize)
    if permute:
        P_in = Permutation(torch.randperm(d_in), dtype=A.dtype)
        P_out = Permutation(torch.randperm(d_out), dtype=A.dtype)
        A = P_in @ A @ P_out
    return CoLALayer(A, bias=bias)

def do_btt_init(A, zero_init, **_):
    for idx, M in enumerate(A.Ms):
        M.d_in, M.d_out = M.shape[0], M.shape[1]
        M.init_std = np.sqrt(min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))
        cola_init(M, zero_init and idx == 1)


def do_bmm0_init(A, vec, zero_init, **_):
    core_dims = get_core_bmm_dims(vec, first_core=0)
    for idx, M in enumerate(A.Ms):
        M.d_in, M.d_out = core_dims[idx]
        M.init_std = np.sqrt(min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))
        cola_init(M, zero_init and idx == 1)


def do_bmm1_init(A, vec, zero_init, **_):
    core_dims = get_core_bmm_dims(vec, first_core=1)
    for idx, M in enumerate(A.Ms):
        M.d_in, M.d_out = core_dims[idx]
        M.init_std = np.sqrt(min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))
        cola_init(M, zero_init and idx == 1)

INITS = {"bmm0": do_bmm0_init, "bmm1": do_bmm1_init}


def init_btt_cores(d_in, d_out, tt_rank, tt_dim, zero_init):
    assert tt_rank > 0, 'tt_rank must be positive'
    ns, ms = tuple(factorize(d_out, tt_dim)), tuple(factorize(d_in, tt_dim))
    rs = (1, tt_rank, 1)
    shapes = (rs, ms, ns)
    cores = []
    print(f'TT shape: {ms} -> {ns}')
    for idx in range(tt_dim):
        size = ns[:idx] + ms[idx + 1:] + (rs[idx] * ms[idx], rs[idx + 1] * ns[idx])
        core = torch.randn(*size, dtype=torch.float32)
        core.d_in = rs[idx] * ms[idx]
        core.d_out = ns[idx] * rs[idx + 1]
        core.init_std = np.sqrt(min(core.d_in, core.d_out) * core.d_out / (core.d_out * core.d_in * core.d_in))
        cola_init(core, zero_init and idx == tt_dim - 1)
        cores.append(core)
    return cores, shapes


def build_low_rank_moe(d_in, d_out, num_experts, bias=True, zero_init=False, num_active_experts=2, **_):
    # each expert is a BTT layer
    experts = [
        build_low_rank_norm(d_in, d_out, rank_frac=1 / num_experts, bias=bias, zero_init=zero_init) for _ in range(num_experts)
    ]
    return CoLAMoELayer(experts, k=num_active_experts)


def build_btt_moe(d_in, d_out, num_experts, tt_dim, tt_rank, bias=True, zero_init=False, num_active_experts=2, **_):
    # each expert is a BTT layer
    experts = [build_opt_btt(d_in, d_out, tt_dim, tt_rank, bias=bias, zero_init=zero_init) for _ in range(num_experts)]
    return CoLAMoELayer(experts, k=num_active_experts)


def build_btt_norm_moe(d_in, d_out, num_experts, tt_dim, tt_rank, bias=True, zero_init=False, num_active_experts=2, **_):
    # each expert is a BTT layer
    experts = [
        build_opt_btt(d_in, d_out, tt_dim, tt_rank, bias=bias, zero_init=zero_init, normalize=True) for _ in range(num_experts)
    ]
    return CoLAMoELayer(experts, k=num_active_experts)


def build_dense_moe(d_in, d_out, num_experts, bias=True, zero_init=False, num_active_experts=2, **_):
    experts = [build_dense(d_in, d_out, bias=bias, zero_init=zero_init) for _ in range(num_experts)]
    return CoLAMoELayer(experts, k=num_active_experts)


def build_einsum(d_in, d_out, expr, init_type='bmm0', do_sgd_lr=False, bias=True, zero_init=False, normalize=False,
                             do_rms_norm=False, **_):
    α, β, γ, δ, ε, φ, _ = vec = construct_vec_from_exps(d_in, d_out, expr, procedure="round")  # integer array of dimensions
    if (β == 1) and (δ == 1):
        return build_btt_vec(d_in, d_out, expr, init_type=init_type, do_sgd_lr=do_sgd_lr, bias=bias, zero_init=zero_init,
                             normalize=normalize, do_rms_norm=do_rms_norm)
    cores = gen_cores(vec)  # random initalized cores
    first_ein_expr, second_ein_expr = get_einsum_exprs(vec)  # einsum string
    A = EinOpVec2(cores, first_ein_expr, second_ein_expr, shapes=((γ, β, α), (φ, ε, δ)), normalize=normalize)
    INITS[init_type](A=A, vec=vec, do_sgd_lr=do_sgd_lr, d_in=d_in, d_out=d_out, zero_init=zero_init)
    return CoLALayer(A, bias=bias, in_features=d_in, out_features=d_out, padding=True, do_rms_norm=do_rms_norm)


def build_btt_norm_moe_parallel(d_in, d_out, num_experts, tt_dim, tt_rank, bias=True, zero_init=False, num_active_experts=2, **_):
    assert tt_rank == 1, 'Parallel BTT MoE experts should have rank 1. Why would you want more?'
    btt_layer = build_opt_btt(d_in, d_out, tt_dim, tt_rank=num_experts, bias=bias, zero_init=zero_init, normalize=True)
    # redo init and lr_mult by pretending they are rank 1
    A, B, *_ = btt_layer.A.Ms
    A.d_out /= num_experts
    B.d_in /= num_experts
    cola_init(A, zero_init=False)
    cola_init(B, zero_init=zero_init)
    return BTTMoELayer(btt_layer, k=num_active_experts)

def build_btt_vec(d_in, d_out, expr, init_type='bmm0', do_sgd_lr=False, bias=True, zero_init=False, normalize=False, **_):
    α, β, γ, δ, ε, φ, ρ = vec = construct_vec_from_exps(d_in, d_out, expr, procedure="round")
    assert (β == 1) and (δ == 1), f"Not a BTT vec as d_β = {β} and d_δ = {δ}"
    A, B = torch.randn(γ, α, φ * ρ), torch.randn(φ, γ * ρ, ε)
    Op = OptBlockTT([A, B], shapes=((1, ρ, 1), (α, γ), (φ, ε)), normalize=normalize)
    print(f"({d_in}, {d_out}) | {Op.shape}")
    INITS[init_type](A=Op, vec=vec, do_sgd_lr=do_sgd_lr, d_in=d_in, d_out=d_out, zero_init=zero_init)
    return CoLALayer(Op, bias=bias, in_features=d_in, out_features=d_out, padding=True, padding_tol=0.15)


build_fns = {
    'btt': lambda *args, **kwargs: build_opt_btt(*args, init='spect', **kwargs),
    'btt_norm': lambda *args, **kwargs: build_opt_btt(*args, normalize=True, **kwargs),
    'low_rank_moe': build_low_rank_moe,
    'btt_moe': build_btt_moe,
    'btt_norm_moe': build_btt_norm_moe,
    'btt_norm_moe_para': build_btt_norm_moe_parallel,
    'dense_moe': build_dense_moe,
    'dense': build_dense,
    'einsum_norm': lambda *args, **kwargs: build_einsum(*args, normalize=True, **kwargs),
    'einsum': build_einsum,
    'none': None,
}


layer_select_fns = {
    'all': lambda *_: True,
    'none': lambda *_: False,
    'all_but_last': lambda name, i, n: i < n - 1,
    'intermediate': lambda name, i, n: i > 0 and i < n - 1,
}


def zero_init_fn(weight, name):
    return hasattr(weight, 'zero_init') and weight.zero_init


def cola_parameterize(model_builder, base_config, lr, target_config=None, struct='none', layer_select_fn='all_but_last',
                      zero_init_fn=zero_init_fn, extra_lr_mult_fn=lambda p_name: 1, device='cuda', cola_kwargs={},
                      optim_kwargs={}, use_wrong_mult=False):
    """
    Create a model and its optimizer in an appropriate parameterization.
    Takes care of lr adjustment both for scaling up model size and for switching to CoLA layers.

    Usage:
        1. Regular μP: struct == 'none'
        2. Regular μP + dense layers initialized by CoLA: struct == 'dense' (e.g. specify custom zero inits)
        3. Regular μP + switching to arbitrary CoLA layers: struct == 'btt', 'tt', etc.

    Assumes:
    1. Classification head has been zero initialized.
    2. Every parameter tensor with more > 2 axes has been annotated with an attribute "fan_in_dims",
        a tuple of dimensions that are considered to be fan-in dimensions.
    3. If struct == 'none', init scale for the dense model is automatically in μP (often true with standard inits).
    4. layer_select_fn does not select the last layer.
    5. If struct != 'none', zero_init_fn selects the last linear layer in every residual block.

    Args:
        model_builder: function that builds the model
        base_config: kwargs for base model
        lr: learning rate for base model
        input_shape: shape of input model
        target_config: kwargs for scaled up model, same as base_config if not specified.
        init_mult: rescale initialization of all matrix-like parameters
        struct: structure of CoLA layers
        layer_select_fn: function that selects which layers to replace with CoLA layers
        zero_init_fn: function maps linear.weight and module name to whether to zero initialize
        extra_lr_mult_fn: function that maps parameter names to extra lr multipliers
        device: device to put model on
        cola_kwargs: kwargs for building CoLA layers
        optim_kwargs: kwargs for optimizer (AdamW)
    """
    base_model = deferred_init(model_builder, **base_config)
    base_param_shapes = [p.shape for p in base_model.parameters()]
    del base_model
    if target_config is None:
        target_config = base_config

    model = deferred_init(model_builder, **target_config)
    lr_mults = append_lr_mults_to_params(model, base_param_shapes=base_param_shapes, optim_kwargs=optim_kwargs)
    if struct != 'none':
        replace_layers_with_cola_layers(model, struct, cola_kwargs, optim_kwargs, layer_select_fn, zero_init_fn, use_wrong_mult)
    materialize_model(model, lr_mults)
    model.to(device)
    optimizer = adjust_lr_and_create_optimizer(model.named_parameters(), lr, extra_lr_mult_fn, optim_kwargs)
    return model, optimizer


def compute_lr_mult(d_in_old, d_out_old, d_in_new, d_out_new, optim_kwargs):
    # ## TODO: add logic to set struct_lr_mult to 1 if using RSGD ###
    opt_name = optim_kwargs["opt_name"] if "opt_name" in optim_kwargs else "AdamW"
    assert opt_name.lower() in ['adam', 'adamw', 'sgd'], f'Unknown optimizer {opt_name}'
    if 'adam' in opt_name.lower():
        # LR ~ 1 / d_in
        return d_in_old / d_in_new
    else:  # SGD
        # LR ~ d_out / d_in
        return (d_out_new / d_in_new) / (d_out_old / d_in_old)


def append_lr_mults_to_params(model, base_param_shapes, optim_kwargs):
    params, param_names, param_shapes = [], [], []
    for name, p in model.named_parameters():
        params.append(p)
        param_names.append(name)
        param_shapes.append(p.shape)
    # compute μP lr multipliers
    assert len(base_param_shapes) == len(param_shapes), 'Base model and target model have different number of param tensors'
    lr_mults = {}
    for base_shape, name, shape, p in zip(base_param_shapes, param_names, param_shapes, params):
        if base_shape == shape:
            mult = 1
        else:
            # special cases, e.g. word embeddings, positional embeddings
            if hasattr(p, 'fan_in_dims'):
                assert hasattr(p, 'fan_out_dims'), f'Parameter {name} has fan_in_dims but not fan_out_dims.'
                fan_in_dims = p.fan_in_dims
                fan_out_dims = p.fan_out_dims
                d_in_base = prod([base_shape[i] for i in fan_in_dims])
                d_in = prod([shape[i] for i in fan_in_dims])
                d_out_base = prod([base_shape[i] for i in fan_out_dims])
                d_out = prod([shape[i] for i in fan_out_dims])
                mult = compute_lr_mult(d_in_base, d_out_base, d_in, d_out, optim_kwargs)
            # matrix like (d_out, d_in)
            elif len(base_shape) == 2:
                d_out_base, d_in_base = base_shape[0], base_shape[1]
                d_out, d_in = shape[0], shape[1]
                mult = compute_lr_mult(d_in_base, d_out_base, d_in, d_out, optim_kwargs)
            # vector like (d_out,), e.g. bias, layernorm gamma
            elif len(base_shape) == 1:
                mult = 1
            else:
                raise ValueError(
                    f'Non matrix or vector parameter {name} has shape {shape}, but does not have fan_in_dims attribute.')
        p.lr_mult = mult
        if hasattr(p, 'depth_lr_mult'):
            print(f'{name} depth-µP lr mult: {p.depth_lr_mult}')
        lr_mults[name] = mult
    return lr_mults


def replace_layers_with_cola_layers(model, struct, cola_kwargs, optim_kwargs, layer_select_fn, zero_init_fn,
                                    use_wrong_mult=False):
    if isinstance(struct, list):
        build_cola = build_fns[struct[0]]
    else:
        build_cola = build_fns[struct]
    if 'lm_head_struct' in cola_kwargs and cola_kwargs['lm_head_struct']:
        build_head = build_fns[cola_kwargs['lm_head_struct']]
        head_kwargs = cola_kwargs.copy()
        head_kwargs['tt_rank'] = cola_kwargs['lm_head_tt_rank']
        head_kwargs['rank_frac'] = cola_kwargs['lm_head_rank_frac']
    else:
        build_head = build_cola
        head_kwargs = cola_kwargs
    if isinstance(layer_select_fn, str):
        layer_select_fn = layer_select_fns[layer_select_fn]
    assert callable(layer_select_fn), f'layer_select_fn must be callable, got {layer_select_fn}'
    # Create a list of all linear layers and their names
    linear_layers = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    num_layers = len(linear_layers)
    for layer_idx, (name, module) in enumerate(linear_layers):
        if layer_select_fn(name, layer_idx, num_layers):
            if hasattr(module, 'leave_as_dense') and module.leave_as_dense:
                continue
            d_in, d_out = module.in_features, module.out_features
            bias = module.bias is not None
            zero_init = zero_init_fn(module.weight, name)
            if zero_init:
                print(f'Zero init: {name}')
            assert hasattr(module.weight, 'lr_mult'), 'Weights in linear layer must have lr_mult attribute'
            dense_lr_mult = module.weight.lr_mult

            learn_gamma = True
            qkv_names = ["attn.c_proj", "c_attn_k", "c_attn_q", "c_attn_v"]
            if 'do_qk_ln' in cola_kwargs and cola_kwargs['do_qk_ln']:
                # when doing qk LN, don't learn gamma for W_Q, W_K
                qk_layer_names = ['c_attn_q', 'c_attn_k', 'to_q', 'to_k']
                learn_gamma = not any(qk_layer_name in name for qk_layer_name in qk_layer_names)
                if not learn_gamma:
                    print(f'Not learning gamma for {name} due to qk ln')
            if name == 'lm_head':
                cola_layer = build_head(d_in, d_out, bias=bias, zero_init=zero_init, **head_kwargs,
                                        learn_gamma=learn_gamma)  # cola specific lr mult are attached
            elif (any(qkv_name in name for qkv_name in qkv_names)) and hasattr(module, 'use_head_btt') and module.use_head_btt:
                n_head, d_head, head_btt_case = module.n_head, module.d_head, module.head_btt_case
                cola_layer = build_head_btt(n_head, d_head, head_btt_case, bias=bias, zero_init=zero_init, **cola_kwargs,
                                            learn_gamma=learn_gamma)
            elif isinstance(struct, list):
                build_cola = build_fns[struct[layer_idx]]
                cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init, cola_kwargs=cola_kwargs[layer_idx],
                                        learn_gamma=learn_gamma)
            else:
                cola_layer = build_cola(d_in, d_out, bias=bias, zero_init=zero_init, **cola_kwargs,
                                        learn_gamma=learn_gamma)  # cola specific lr mult are attached

            for p in cola_layer.matrix_params:
                if use_wrong_mult:
                    p.lr_mult = dense_lr_mult
                else:
                    if p.dim() == 0:
                        p.lr_mult = 1  # scalar have O(1) lr
                    else:
                        # ## TODO: add logic to set struct_lr_mult to 1 if using RSGD ###
                        if hasattr(p, 'lr_mult'):
                            struct_lr_mult = p.lr_mult
                        else:
                            struct_lr_mult = compute_lr_mult(d_in, d_out, p.d_in, p.d_out, optim_kwargs) / cola_layer.num_mats
                        p.lr_mult = struct_lr_mult * dense_lr_mult  # final cola mult = cola mult * dense mult
            # Split the name to get parent module and attribute name
            name_split = name.rsplit('.', 1)
            # If it's a top-level module
            if len(name_split) == 1:
                setattr(model, name_split[0], cola_layer)
            else:
                parent_name, attr_name = name_split
                parent_module = reduce(getattr, parent_name.split('.'), model)
                setattr(parent_module, attr_name, cola_layer)


def materialize_model(model, lr_mults):
    materialize_module(model)
    # materialization gets rid of lr_mults of those tensors, so we need to add them back
    for name, p in model.named_parameters():
        if not hasattr(p, 'lr_mult'):
            if hasattr(p, 'override_lr'):
                pass
            else:
                p.lr_mult = lr_mults[name]


def adjust_lr_and_create_optimizer(named_parameters, lr, extra_lr_mult_fn, optim_kwargs):
    wd = optim_kwargs['weight_decay'] if 'weight_decay' in optim_kwargs else 0.0
    device_type = optim_kwargs.pop("device_type") if "device_type" in optim_kwargs else "cuda"
    param_groups = []
    for name, param in named_parameters:
        if hasattr(param, 'override_lr'):
            param.lr_mult = param.override_lr / lr
        assert hasattr(param, 'lr_mult'), f'lr_mult not found for {name}'
        mult = param.lr_mult
        extra_mult = extra_lr_mult_fn(name)
        if extra_mult != 1:
            print(f'{name}: {extra_mult} * {mult}')
            mult *= extra_mult
        else:
            print(f'{name}: {mult}')
        if param.dim() >= 2 and param.requires_grad:
            adjusted_wd = wd
        else:
            print(f'No wd for {name}')
            adjusted_wd = 0.0
        adjusted_lr = lr * mult
        param_groups.append({'params': param, 'lr': adjusted_lr, 'weight_decay': adjusted_wd})
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    opt_name = optim_kwargs.pop("opt_name") if "opt_name" in optim_kwargs else "AdamW"
    if opt_name == "AdamW":
        optimizer = AdamW(param_groups, **optim_kwargs, **extra_args)
    elif opt_name == "sgd":
        if "fused" in extra_args.keys():
            extra_args.pop("fused")
        optimizer = SGD(param_groups, **optim_kwargs, **extra_args)
    else:
        optimizer = Adam(param_groups, **optim_kwargs, **extra_args)
    return optimizer


def get_model_summary_and_flops(model, fake_input):
    print('Model:')
    stats = summary(model, input_data=fake_input)
    cola_params = stats.trainable_params
    print(f'Params: {cola_params / 1e6:.2f}M')
    cola_flops = FlopCountAnalysis(model, fake_input).set_op_handle(**custom_ops).total()
    print(f'FLOPs: {cola_flops / 1e6:.2f}M')
    print('=' * 90)
    info = {'cola_params': cola_params, 'cola_flops': cola_flops}

    return info


def btt_flop_count(inputs, outputs):
    if len(inputs) == 3:
        x, W1, W2 = inputs
        batch_size = get_shape(x)[0]
        flops = get_numel(W1) + get_numel(W2)
    elif len(inputs) == 5:
        # for btt_norm_moe_para, which uses more FLOPs than it needs due to naive parallelization
        x, W1, W2, gate, num_active = inputs
        num_experts = get_shape(gate)[1]
        num_active_experts = get_shape(num_active)[0]
        batch_size = get_shape(x)[0]
        flops = get_numel(W1) + get_numel(W2)
        flops = flops * (num_active_experts / num_experts)
    else:
        raise ValueError(f'Unexpected number of inputs: {len(inputs)}')
    return batch_size * flops


def scaled_dot_product_attention_flop_count(inputs, outputs):
    output_shape = get_shape(outputs[0])  # ([batch dims], seq_len, head_dim)
    B = prod(output_shape[:-2])  # batch suze
    L, D = output_shape[-2], output_shape[-1]  # (seq_len, head_dim)
    qk_flops = B * (L * D * L)  # (L, D) @ (D, L)
    v_flops = B * (L * L * D)  # (L, L) @ (L, D)
    return qk_flops + v_flops


def get_shape(x):
    return x.type().sizes()


def get_numel(x):
    return prod(get_shape(x))


def normalized_rms(x):
    d_in, d_out = x.d_in, x.d_out
    rms = torch.sqrt(torch.mean(x**2) + 1e-8).item()
    expected = np.sqrt(min(d_in, d_out) * d_out / (d_out * d_in * d_in))
    return rms / expected


def rms(x, dim, eps=1e-16):
    return torch.sqrt(torch.mean(x**2, dim) + eps)


def rms_norm(x, dim, eps=1e-16):
    return x / rms(x, dim, eps).unsqueeze(dim)


def update_singular_vectors(model, scale_grad_by=1):
    for name, module in model.named_modules():
        if isinstance(module, CoLALayer):
            if hasattr(module, 'top_singular_vec'):
                # v = module.top_singular_vec
                dv = module.top_singular_vec.grad * scale_grad_by  # proportional to A^T @ A @ v
                v_next = rms_norm(dv * module.out_features, dim=-1)
                if rms(v_next, dim=-1) < 0.5:
                    # resample v
                    v_next = torch.randn_like(v_next)
                module.top_singular_vec.data = v_next
                # zero out the gradient
                module.top_singular_vec.grad = None


def custom_einsum_flop_count(inputs, outputs):
    """
    Count flops for the einsum operation.
    Adapted from implementation in https://github.com/facebookresearch/fvcore
    """
    # Inputs of einsum should be a list of length 2+.
    # Inputs[0] stores the equation used for einsum.
    # Inputs[1] stores the list of input shapes.
    # Inputs[2] optionally stores the optimized path of contraction.
    assert len(inputs) >= 2, len(inputs)
    equation = inputs[0].toIValue()
    # Get rid of white space in the equation string.
    equation = equation.replace(" ", "")
    input_shapes_jit = inputs[1].node().inputs()
    input_shapes = [get_shape(v) for v in input_shapes_jit]

    # Re-map equation so that same equation with different alphabet
    # representations will look the same.
    letter_order = OrderedDict((k, 0) for k in equation if k.isalpha()).keys()
    mapping = {ord(x): 97 + i for i, x in enumerate(letter_order)}
    equation = equation.translate(mapping)

    if equation == "abc,abd->acd":
        n, c, t = input_shapes[0]
        p = input_shapes[-1][-1]
        flop = n * c * t * p
        return flop

    elif equation == "abc,adc->adb":
        n, t, g = input_shapes[0]
        c = input_shapes[-1][1]
        flop = n * t * g * c
        return flop
    else:
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = opt_einsum.contract_path(equation, *np_arrs, optimize='optimal')[1]
        return int(optim.opt_cost) / 2


custom_ops = {
    'prim::PythonOp.BlockTeTr': btt_flop_count,
    'aten::scaled_dot_product_attention': scaled_dot_product_attention_flop_count,
    'aten::einsum': custom_einsum_flop_count,
}
