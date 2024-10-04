from math import prod
from cola.ops.operator_base import LinearOperator
import torch
from torch.nn import functional as F
from mm.btt_mvm import btt_mvm, btt_mvmt

class EinOpVec2(LinearOperator):
    def __init__(self, Ms, first_ein_expr, second_ein_expr, shapes, normalize=False):
        self.normalize = normalize
        self.Ms = Ms
        self.first_ein_expr = add_batch_to_einsum_expr(first_ein_expr)
        self.first_ein_expr_rev = add_batch_to_einsum_expr(reverse_einsum_expr(first_ein_expr))
        self.second_ein_expr = add_batch_to_einsum_expr(second_ein_expr)
        self.second_ein_expr_rev = add_batch_to_einsum_expr(reverse_einsum_expr(second_ein_expr))
        self.in_shape = tuple([sh for sh in shapes[0] if sh > 1])
        self.out_shape = tuple([sh for sh in shapes[1] if sh > 1])
        super().__init__(dtype=Ms[0].dtype, shape=(prod(shapes[0]), prod(shapes[1])))

    def get_normalized_cores(self):
        if not self.normalize:
            return self.Ms
        else:
            normalized_Ms = []
            for M in self.Ms:
                rms = torch.sqrt(torch.mean(M**2) + 1e-6)
                max_rms = (min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))**0.5
                denom = max(1, rms / max_rms)
                M_normalized = M / denom
                normalized_Ms.append(M_normalized)
            return normalized_Ms

    def _rmatmat(self, X):
        bsz = X.shape[0]
        Ms = self.get_normalized_cores()
        X = X.reshape((bsz, ) + self.in_shape).contiguous()
        Z = torch.einsum(self.first_ein_expr, X, Ms[0]).contiguous()
        Y = torch.einsum(self.second_ein_expr, Z, Ms[1]).contiguous()
        Y = Y.reshape(bsz, -1)
        return Y


def reverse_einsum_expr(expr):
    rev = expr.replace("->", ",").split(",")[::-1]
    rev = ",".join(rev[:-1]) + "->" + rev[-1]
    return rev


def add_batch_to_einsum_expr(expr):
    loc = expr.find(">") + 1
    out = f"Z{expr[:loc]}Z{expr[loc:]}"
    return out

class OptBlockTT(LinearOperator):
    """
   Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms, shapes, normalize=False):
        self.Ms = Ms
        self.normalize = normalize
        dtype = self.Ms[0].dtype
        self.rs, self.ms, self.ns = shapes
        shape = (prod(self.ms), prod(self.ns))
        d_in, d_out = shape
        super().__init__(dtype, shape)

    def _rmatmat(self, v, gate=None, num_active=None):  # x -> x @ A
        # gate is a sparse vector for MoE
        W0, W1 = self.Ms[0], self.Ms[1]
        if self.normalize:
            rms_W0 = torch.sqrt(torch.mean(W0**2.) + 1e-8)
            d_in, d_out = W0.d_in, W0.d_out
            max_rms0 = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
            if len(self.Ms) > 2:
                W0 = self.Ms[2] * W0 / max(1, rms_W0 / max_rms0)
            else:
                W0 = W0 / max(1, rms_W0 / max_rms0)

            rms_W1 = torch.sqrt(torch.mean(W1**2.) + 1e-8)
            d_in, d_out = W1.d_in, W1.d_out
            max_rms1 = (min(d_in, d_out) * d_out / (d_out * d_in * d_in))**0.5
            if len(self.Ms) > 2:
                W1 = self.Ms[3] * W1 / max(1, rms_W1 / max_rms1)
            else:
                W1 = W1 / max(1, rms_W1 / max_rms1)
        if gate is not None:
            active = (gate > 0).sum(-1)
            assert (active == num_active).float().mean() > 0.95, f"Expected {num_active} active experts, got {active}"
        # last argument is for keeping track of active experts in the flop counter
        num_active = torch.empty(num_active) if num_active is not None else None
        return btt_mvm(v, W0, W1, (self.rs, self.ms, self.ns), gate, num_active)

    def _rmatmat_T(self, v):  # g -> g @ A^T
        return btt_mvmt(v, self.Ms[0], self.Ms[1], (self.rs, self.ms, self.ns))

    def __str__(self):
        return "BlockTT"
