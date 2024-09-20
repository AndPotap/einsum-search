from collections import Counter
from math import ceil
import torch
from torch.nn import Linear
import numpy as np
from sympy import factorint


def to_dense_gbtt_flat(F, shapes):
    α, _, γ, _, ε, φ, ρ, ξ = shapes
    shape0 = γ * ξ * α * φ * ρ
    A, B = F[:shape0], F[shape0:]
    A = A.reshape(γ, ξ, α, φ, ρ)
    B = B.reshape(φ, ξ, γ, ρ, ε)
    aux0 = A.permute((1, 0, 3, 2, 4))  # (ξ, γ, φ, α, ρ)
    aux1 = B.permute((1, 2, 0, 3, 4))  # (ξ, γ, φ, ρ, ε)
    out = aux0 @ aux1  # (ξ, γ, φ, α, ε)
    out = out.permute((0, 3, 1, 4, 2))
    out = out.reshape(ξ, α * γ, ε * φ)
    aux = torch.unbind(out, dim=0)
    out = torch.block_diag(*aux)
    return out


def to_dense_btt_flat(F, shapes):
    α, γ, ε, φ, ρ = shapes
    shape0 = γ * α * φ * ρ
    A, B = F[:shape0], F[shape0:]
    W = to_dense_btt(A, B, shapes)
    return W.reshape(-1)


def to_dense_btt(A, B, shapes):
    α, γ, ε, φ, ρ = shapes
    A = A.reshape(γ, α, φ, ρ)
    B = B.reshape(φ, γ, ρ, ε)
    aux0 = A.permute((0, 2, 1, 3))  # (γ, φ, α, ρ)
    aux1 = B.permute((1, 0, 2, 3))  # (γ, φ, ρ, ε)
    out = aux0 @ aux1  # (γ, φ, α, ε)
    out = out.permute((2, 0, 3, 1))
    out = out.reshape(α * γ, ε * φ)
    return out


def to_dense_einsum(A, B):
    φ, δ, ρ, γ, α = A.shape  # (φ, δ, ρ, γ, α)
    _, ε, _, _, β = B.shape  # (φ, ε, ρ, γ, β)
    aux0 = A.permute((3, 0, 4, 1, 2))  # (γ, φ, α, δ, ρ)
    aux0 = aux0.reshape(γ, φ, α * δ, ρ)
    aux1 = B.permute((3, 0, 2, 4, 1))  # (γ, φ, ρ, β, ε)
    aux1 = aux1.reshape(γ, φ, ρ, β * ε)
    out = aux0 @ aux1
    out = out.reshape(γ, φ, α, δ, β, ε)
    out = out.permute((0, 4, 2, 1, 5, 3))
    out = out.reshape(γ * β * α, φ * ε * δ)
    return out


def get_core_bmm_dims(vec, first_core=0):
    if len(vec) == 7:
        α, β, γ, δ, ε, φ, ρ = vec
        if first_core == 0:
            d0_in = α
            d0_out = δ * φ * ρ
            d1_in = β * γ * ρ
            d1_out = ε
        else:
            d0_in = α * γ * ρ
            d0_out = δ
            d1_in = β
            d1_out = ε * φ * ρ
        return (d0_in, d0_out), (d1_in, d1_out)
    elif len(vec) == 8:
        α, _, γ, _, ε, φ, ρ, ξ = vec
        if first_core == 0:
            d0_in = α
            d0_out = φ * ρ
            d1_in = γ * ρ
            d1_out = ε
        return (d0_in, d0_out), (d1_in, d1_out)
    elif len(vec) == 10:
        d0_in = vec[0]
        d0_out = vec[3] * vec[6] * vec[7] * vec[8]
        d1_in = vec[1] * vec[6]
        d1_out = vec[4] * vec[8]
        d2_in = vec[2] * vec[7] * vec[8] * vec[9]
        d2_out = vec[5]
        return (d0_in, d0_out), (d1_in, d1_out), (d2_in, d2_out)
    else:
        raise ValueError(f"Vec is not either 7 or 10, it is {len(vec)}")


def select_factorizer(name):
    if name == "up":
        return FactUp
    elif name == "upvec":
        return FactUpVec
    elif name == "bttvec":
        return FactUpBTT
    elif name == "btt3vec":
        return FactBTT3Vec
    elif name == "gbtt_round":
        return FactGBTT()
    else:
        # raise ValueError(f"Factorizer: {name} not found")
        return None


class FactGBTT:
    def __init__(self, text_exps):
        self.set_exps(text_exps)

    def set_exps(self, text_exps):
        self.exps = [float(c) for c in text_exps[1:-1].split("|")]
        assert len(self.exps) == 8, f"Not enought exps for gbtt, requires 8 but {len(self.exps)} were give"
        α, β, γ, δ, ε, φ, _, ξ = self.exps
        assert (β == 0.0) and (δ == 0.0), f"Not in the BTT family as β={β} and δ={δ}"
        assert abs(α + β + γ + ξ - 1) <= 1e-2, "x exponents summing more than 1"
        assert abs(δ + ε + φ + ξ - 1) <= 1e-2, "y exponents summing more than 1"

    def construct_vec(self, d_in, d_out):
        α, _, γ, _, ε, φ, ρ, ξ = self.exps
        shared_size = factorize_round(min(d_in, d_out), [ξ])[0]
        sizes = [round(d_in / shared_size), min(d_in, d_out), round(d_out / shared_size)]
        x_exps = list(map(lambda x: x / (1 - ξ), [α, 0.0, γ]))
        y_exps = list(map(lambda x: x / (1 - ξ), [0.0, ε, φ]))
        vec = factorize_round(sizes[0], x_exps)
        vec += factorize_round(sizes[2], y_exps)
        vec += factorize_round(sizes[1], [ρ])
        vec += [shared_size]
        return vec


class Factorizer:
    def __init__(self, ρs=[]):
        self.ρs = ρs
        self.exps = []
        self.layers = []

    def sample(self, **_):
        self.exps = []

    def construct_vec(self, d_in, d_out):
        vec = construct_vec(d_in, d_out, self.exps, procedure="round")
        return vec


class FactUp(Factorizer):
    def __init__(self, cores_n, ρs):
        super().__init__()

    def construct_vec(self, d_in, d_out):
        vec = construct_vec(d_in, d_out, self.exps, procedure="up")
        return vec


class FactUpVec(FactUp):
    def sample(self, expr):
        self.exps = get_exps_from_text(expr)


class FactUpBTT(FactUp):
    def sample(self, expr):
        self.exps = gen_btt_coefs(cores_n=self.cores_n, int_pow=self.int_pow)


class FactBTT3Vec:
    def __init__(self, int_pow, **_):
        self.flops = 0
        self.cases = {}
        self.layers = []
        self.padding = True

    def sample(self, expr):
        self.exps = get_exps_from_text(expr)

    def Linear(self, in_features, out_features, bias):
        vec = self.construct_vec(in_features, out_features)
        d_in = np.prod(factorize_up(in_features, self.exps[0]))
        d_out = np.prod(factorize_up(out_features, self.exps[-1]))
        self.cases[(d_in, d_out)] = vec
        return Linear(d_in, d_out, bias=bias)

    def construct_vec(self, d_in, d_out):
        if (d_in, d_out) in self.cases.keys():
            return self.cases[(d_in, d_out)]
        else:
            facs = [factorize_up(siz, self.exps[idx]) for idx, siz in enumerate([d_in, d_in, d_out])]
            vec = [*facs[0], *facs[-1], *facs[1]]
            return vec

    @staticmethod
    def get_cores(vec):
        cores = []
        for idx in range(3):
            core = torch.randn(vec[idx], vec[idx + 1], vec[idx + 2], vec[idx + 3])
            cores.append(core)
        return cores

    def register_info(self, d_in, d_out, ein_expr, vec):
        flops = compute_flops_einsum(vec)
        self.flops += flops
        info = {"d_in": d_in, "d_out": d_out, "ein_expr": ein_expr, "flops": flops}
        self.layers.append(info)

    def get_unique_ein_expr(self):
        expr_counts = Counter([la["ein_expr"] for la in self.layers])
        expr_counts = dict(sorted(expr_counts.items(), key=lambda item: item[1])[::-1])
        return expr_counts

    def log_data(self):
        aux = [*self.exps[0], *self.exps[-1], *self.exps[1]]
        data = {"theta": "|".join([str(val) for val in aux])}
        return data


def factorize(x, n):
    prime_factors = factorint(x)
    numbers = [1] * n
    for prime, count in prime_factors.items():
        for _ in range(count):
            min_index = min(range(n), key=lambda i: numbers[i])
            numbers[min_index] *= prime
    return sorted(numbers)


def compute_flops_einsum(vec):
    if len(vec) == 7:
        α, _, _, _, ε, _, _ = vec
        flops = np.prod(vec) / ε + np.prod(vec) / α
        return flops
    elif len(vec) == 10:
        flops = np.prod(vec)
        flops *= (1 / (vec[4] * vec[5]) + 1 / (vec[0] * vec[5]) + 1 / (vec[0] * vec[1]))
        return flops
    else:
        raise ValueError(f"Vec not of size 7 or size 10, {len(vec):,d}")


def construct_vec_from_exps(d_in, d_out, text_exps, procedure="round"):
    exps = get_exps_from_text(text_exps)
    vec = construct_vec(d_in, d_out, exps, procedure=procedure)
    return vec


def get_exps_from_text(expr: str) -> list[float]:
    all = [float(c) for c in expr[1:-1].split("|")]
    if len(all) == 7:
        α, β, γ, δ, ε, φ, ρ = all
        exps = [[α, β, γ], [δ, ε, φ], [ρ]]
    elif len(all) == 10:
        exps = [all[0:3], all[6:], all[3:6]]
    else:
        raise ValueError(f"Expr {expr} has len: {len(all)}")
    return exps


def construct_vec(d_in: int, d_out: int, exps: list[list[float]], procedure: str) -> list[int]:
    assert procedure in ["opt", "up", "round"], f"Factorization procedure: {procedure} not found"
    assert abs(sum(exps[0]) - 1) <= 1e-2, f"Input coefficients must sum to 1, got: {sum(exps[0])}"
    assert abs(sum(exps[1]) - 1) <= 1e-2, f"Output coefficients must sum to 1, got: {sum(exps[-1])}"
    sizes = [d_in, d_out, min(d_in, d_out)]
    fact_fn = {"up": factorize_up, "round": factorize_round}[procedure]
    vec = []
    for size, exp in zip(sizes, exps):
        vec += fact_fn(size, exp)
    return vec


def factorize_round(M, exps):
    Ms = M**np.array(exps)
    out = [round(mdx) for mdx in Ms]
    return out


def factorize_up(M: int, exps: np.ndarray):
    Ms = M**np.array(exps)
    out = [ceil(mdx) for mdx in Ms]
    return out


def gen_btt_coefs(cores_n: int, int_pow: list[float]) -> list[list[float]]:
    assert cores_n == 4, f"functionality not available for cores_n = {cores_n}"
    theta_alpha = np.random.uniform(low=0.0, high=1.0, size=1)[0]
    theta_alpha = theta_alpha if theta_alpha >= 0.1 else 0.
    theta_alpha = theta_alpha if theta_alpha <= 0.9 else 1.
    theta_delta = 1. - theta_alpha

    theta_phi = np.random.uniform(low=0.0, high=1.0, size=1)[0]
    theta_phi = theta_phi if theta_phi >= 0.1 else 0.
    theta_phi = theta_phi if theta_phi <= 0.9 else 1.
    theta_gamma = 1. - theta_phi

    int_pow = np.random.choice(a=int_pow, size=1)

    exps = [[0.0, theta_phi, theta_gamma], int_pow, [theta_alpha, 0.0, theta_delta]]
    return exps


def gen_partition_coeffs(input_size, inter_size, output_size, int_pow):
    in_w = sample_stick_break(alpha=1, num_samples=input_size)
    # inter_w = np.random.uniform(low=0.0, high=int_pow, size=inter_size)
    inter_w = np.random.choice(a=int_pow, size=1)
    out_w = sample_stick_break(alpha=1, num_samples=output_size)
    return in_w, inter_w, out_w


def sample_stick_break(alpha, num_samples):
    thetas = np.zeros(shape=(num_samples, ))
    betas = np.random.beta(1, alpha, size=num_samples - 1)
    cumprod = np.concatenate((np.ones(1, ), np.cumprod(1 - betas)[:-1]))
    thetas[:num_samples - 1] = cumprod * betas
    thetas[-1] = 1. - np.sum(thetas)
    case = np.random.randint(low=0, high=num_samples, size=1)[0]
    if case > 0:
        locs = np.random.choice(np.arange(len(thetas)), size=case, replace=False)
        thetas[locs] = 0.
    thetas = thetas / np.sum(thetas)
    np.random.shuffle(thetas)
    return thetas


def get_einsum_expr(vec):
    x_str = ["g", "b", "a"]
    A_str = ["f", "d", "r", "g", "a"]
    B_str = ["f", "e", "r", "g", "b"]
    y_str = ["f", "e", "d"]

    exprs = []
    for val in [x_str, A_str, B_str, y_str]:
        exprs.append("".join(val))
    expr = ",".join(exprs[:-1])
    expr = f"{expr}->{exprs[-1]}"

    vec_loc_to_letter = {idx: let for idx, let in enumerate(["a", "b", "g", "d", "e", "f", "r"])}
    let_to_suppress = [vec_loc_to_letter[loc] for loc, v in enumerate(vec) if v == 1]
    for let in let_to_suppress:
        expr = expr.replace(let, "")
    return expr


def gen_cores(vec):
    α, β, γ, δ, ε, φ, ρ = vec
    A = torch.randn(φ, δ, ρ, γ, α).squeeze()
    B = torch.randn(φ, ε, ρ, γ, β).squeeze()
    return A, B


def get_einsum_exprs(vec):
    x_str = ["g", "b", "a"]
    A_str = ["f", "d", "r", "g", "a"]
    z_str = ["f", "d", "r", "g", "b"]
    B_str = ["f", "e", "r", "g", "b"]
    y_str = ["f", "e", "d"]

    first_exprs = []
    for val in [x_str, A_str, z_str]:
        first_exprs.append("".join(val))
    first_expr = ",".join(first_exprs[:-1])
    first_expr = f"{first_expr}->{first_exprs[-1]}"

    second_exprs = []
    for val in [z_str, B_str, y_str]:
        second_exprs.append("".join(val))
    second_expr = ",".join(second_exprs[:-1])
    second_expr = f"{second_expr}->{second_exprs[-1]}"

    vec_loc_to_letter = {idx: let for idx, let in enumerate(["a", "b", "g", "d", "e", "f", "r"])}
    let_to_suppress = [vec_loc_to_letter[loc] for loc, v in enumerate(vec) if v == 1]
    for let in let_to_suppress:
        first_expr = first_expr.replace(let, "")
        second_expr = second_expr.replace(let, "")
    return first_expr, second_expr
