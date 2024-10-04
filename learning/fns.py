from collections import Counter
from math import ceil
import torch
from torch.nn import Linear
import numpy as np
from sympy import factorint

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
