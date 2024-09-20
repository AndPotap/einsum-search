from math import prod
from cola.ops.operator_base import LinearOperator
import torch
from torch.nn import functional as F
from learning.fns import to_dense_btt
from mm.blockdiag_butterfly_multiply import blockdiag_butterfly_multiply
from mm.btt_mvm import btt_mvm, btt_mvmt
from mm.non_btt_mvm import non_btt_mvm
from mm.rbtt_mvm import rbtt_mvm
from mm.gbtt_mvm import gbtt_mvm
from mm.btt_mvm_correct import btt_mvm as cbtt_mvm


class DenseNorm(LinearOperator):
    def __init__(self, M):
        self.M = M
        super().__init__(M.dtype, M.shape)

    def _rmatmat(self, v):
        M = self.M
        max_rms = (min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))**0.5
        rms = torch.sqrt(torch.mean(M**2) + 1e-6)
        denom = max(1, rms / max_rms)
        M_normalized = M / denom
        return v @ M_normalized


class NonBTT(LinearOperator):
    def __init__(self, Ms, shapes, act_fn):
        self.Ms, self.shapes, self.act_fn = Ms, shapes, act_fn
        _, ms, ns = shapes
        super().__init__(dtype=Ms[0].dtype, shape=(prod(ms), prod(ns)))

    def _rmatmat(self, X):
        out = non_btt_mvm(X, self.Ms[0], self.Ms[1], self.shapes, self.act_fn)
        return out


class EinOpVec(LinearOperator):
    def __init__(self, Ms, ein_expr, shapes, normalize=False):
        self.normalize = normalize
        self.Ms = Ms
        self.ein_expr = add_batch_to_einsum_expr(ein_expr)
        self.ein_expr_rev = add_batch_to_einsum_expr(reverse_einsum_expr(ein_expr))
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
        X = X.reshape((bsz, ) + self.in_shape)
        Y = torch.einsum(self.ein_expr, X, *Ms)
        Y = Y.reshape(bsz, -1)
        return Y

    def _matmat(self, X):
        batch_n = X.shape[0]
        Ms = self.get_normalized_cores()
        X = X.reshape((batch_n, ) + self.out_shape)
        X = torch.einsum(self.ein_expr_rev, X, *Ms[::-1])
        X = X.reshape(batch_n, -1)
        return X

    def get_core_prop(self):
        props, sigma_max = compute_contribution(self)
        return props, sigma_max


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


class BlockEinOpVec(LinearOperator):
    def __init__(self, ein_ops):
        self.ein_expr = ein_ops[0].ein_expr
        self.in_shape = ein_ops[0].in_shape
        self.out_shape = ein_ops[0].out_shape
        self.normalize = ein_ops[0].normalize
        self.num_blocks = len(ein_ops)  # k
        # assert all ops have same exprs, shapes, and noramlize, and allow padding
        for ein_op in ein_ops:
            assert ein_op.ein_expr == self.ein_expr
            assert ein_op.in_shape == self.in_shape
            assert ein_op.out_shape == self.out_shape
            assert ein_op.normalize == self.normalize
        self.ein_expr = add_block_to_einsum_expr(self.ein_expr)
        Ms_blocks = [ein_op.Ms for ein_op in ein_ops]  # [(A1, B1, ...), (A2, B2, ...), ..., (Ak, Bk, ...)]
        # transpose to [(A1, A2, ...), (B1, B2, ...), ..., ]
        Ms_blocks = list(zip(*Ms_blocks))
        # stack the Ms, which are assumed to have been properly initialized
        self.Ms = [torch.stack(M_blocks, dim=0) for M_blocks in Ms_blocks]
        assert len(self.Ms) == len(Ms_blocks), "Number of blocks must match"
        for stacked_M, M_blocks in zip(self.Ms, Ms_blocks):
            stacked_M.d_in, stacked_M.d_out = M_blocks[0].d_in, M_blocks[0].d_out
        super().__init__(dtype=self.Ms[0].dtype, shape=ein_ops[0].shape)

    def get_normalized_cores(self):
        if not self.normalize:
            return self.Ms
        else:
            normalized_Ms = []
            for M in self.Ms:
                rms = torch.sqrt(torch.mean(M**2, dim=list(range(1, M.ndim)), keepdim=True) + 1e-6)
                max_rms = (min(M.d_in, M.d_out) * M.d_out / (M.d_out * M.d_in * M.d_in))**0.5
                denom = torch.max(rms / max_rms, torch.ones_like(rms))
                M_normalized = M / denom
                normalized_Ms.append(M_normalized)
            return normalized_Ms

    def _rmatmat(self, X):
        block_dim, batch_dim = X.shape[0], X.shape[1]
        assert block_dim == self.num_blocks, f"Block dimension {block_dim} != {self.num_blocks}"
        Ms = self.get_normalized_cores()
        X = X.reshape((block_dim, batch_dim) + self.in_shape)
        X = torch.einsum(self.ein_expr, X, *Ms)
        X = X.reshape(block_dim, batch_dim, -1)
        return X


def compute_contribution(A):
    with torch.no_grad():
        E = A.to_dense()
        U, S, V = torch.linalg.svd(E)
    sigma_max = S[0]
    v, u = V[0, :], U[:, 0]
    loss = (v @ A) @ u
    loss.backward()
    contr = [torch.linalg.norm(M.grad) for M in A.Ms]
    # contr = [torch.linalg.norm(M.A.grad) for M in A.Ms]
    total = sum(contr)
    props = [gn / total for gn in contr]
    return props, sigma_max


def reverse_einsum_expr(expr):
    rev = expr.replace("->", ",").split(",")[::-1]
    rev = ",".join(rev[:-1]) + "->" + rev[-1]
    return rev


def add_batch_to_einsum_expr(expr):
    loc = expr.find(">") + 1
    out = f"Z{expr[:loc]}Z{expr[loc:]}"
    return out


def add_block_to_einsum_expr(expr):
    factor_exprs = expr.split("->")[0].split(",")
    out_expr = expr.split("->")[1]
    factor_exprs = [f"E{f}" for f in factor_exprs]
    out_expr = f"E{out_expr}"
    return ",".join(factor_exprs) + "->" + out_expr


class EinOp(LinearOperator):
    def __init__(self, Ms, ein_exps, shapes):
        self.Ms = Ms
        self.ein_exps = ein_exps
        self.in_shape = tuple(shapes[0])
        shape = (prod(shapes[0]), prod(shapes[1]))
        super().__init__(dtype=Ms[0].dtype, shape=shape)

    def _rmatmat(self, X):
        batch_dim = X.shape[0]
        X = X.reshape((batch_dim, ) + self.in_shape)
        for ldx, (new_idx, in_idx, out_idx) in enumerate(self.ein_exps):
            exp = f"{new_idx[0]},z{in_idx[0]}->z{out_idx[0]}"
            X = torch.einsum(exp, self.Ms[ldx], X)
        X = X.reshape(batch_dim, -1)
        return X


class BlockDiagWithTranspose(LinearOperator):
    """
    Block-diagonal operator with transpose
    Args:
        M (array_like): Block-diagonal matrix of shape (b, n, m)
        transpose (bool): Whether to transpose n and m after matmul
    """
    def __init__(self, M, transpose):
        dtype = M.dtype
        self.M = M
        self.b, self.n, self.m = M.shape
        shape = (self.b * self.n, self.b * self.m)
        self.transpose_out = transpose
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        # v: (batch, d_in)
        v = v.view(-1, self.b, self.n)  # (i, b, n)
        v = torch.einsum('ibn,bnm->ibm', v, self.M)  # (i, b, m)
        if self.transpose_out:
            v = v.transpose(1, 2)
        v = v.reshape(-1, self.b * self.m)
        return v


class SubvectorsMatch(LinearOperator):
    def __init__(self, M):
        dtype = M.dtype
        self.k, self.n, self.m = M.shape
        shape = (self.n * self.m, self.k * self.n)
        self.M = M

        self.dim_in = self.n * self.m
        self.indices = torch.stack([torch.randperm(self.dim_in) for _ in range(self.k)]).to("cuda")

        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        batch_size, input_dim = v.shape

        assert input_dim == self.dim_in, f"The dimension doesn't match: {input_dim} != {self.dim_in}"

        # Get k copies
        v = v.unsqueeze(1).expand(-1, self.k, -1)

        # Randomly permute the third dimension
        v = v.gather(2, self.indices.expand(batch_size, -1, -1))

        # Segments into sections
        v = v.view(batch_size, self.k, self.n, self.m)

        # Measure similarity
        w = (v * self.M.unsqueeze(0)).sum(dim=-1)

        return w.reshape(batch_size, -1)


class Conv(LinearOperator):
    def __init__(self, filters, shape):
        kernel_size = filters.shape[-1]
        self.filters = filters
        # Check kernel is odd?
        assert kernel_size % 2 == 1, f"kernel size {kernel_size} must not be even"
        self.pad = int((kernel_size - 1) / 2)
        super().__init__(dtype=filters.dtype, shape=shape)

    def _rmatmat(self, V):
        # V.shape = (batch_size, d_in)
        out = F.conv1d(V[:, None, :], self.filters, None, 1, padding=self.pad)
        if self.shape[0] >= self.shape[1]:
            out = out[:, 0, :self.shape[1]]
        else:
            out = F.pad(out[:, 0, :], (0, self.shape[1] - self.shape[0]))
        return out  # (batch_size, d_out)


class Monarch(LinearOperator):
    def __init__(self, Ms, shape):
        self.Ms = Ms
        dtype = self.Ms[0].dtype
        num_blocks = self.Ms[0].shape[0]
        in_blksz, out_blksz = self.Ms[0].shape[-1], self.Ms[-1].shape[-1]
        self.d_in_ext = in_blksz * num_blocks
        self.d_out_ext = out_blksz * num_blocks
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        d_in, d_out = self.shape
        if v.shape[-1] < self.d_in_ext:
            v = F.pad(v, (0, self.d_in_ext - d_in))

        out = blockdiag_butterfly_multiply(v, self.Ms[0], self.Ms[1])

        if out.shape[-1] > d_out:
            out = out[..., :d_out]
        return out


class Diag(LinearOperator):
    def __init__(self, diag):
        self.diag = diag
        super().__init__(diag.dtype, (len(diag), len(diag)))

    def _rmatmat(self, v):
        return self.diag * v


class Banded(LinearOperator):
    def __init__(self, bands, d_in, d_out, max_block_size=32):
        """ LinearOperator of shape (d_in, d_out) with (k, max(d_in, d_out)) matrix of bands."""
        self.k, self.d = bands.shape
        self.d_in, self.d_out = d_in, d_out
        assert (self.k - 1) % 2 == 0, f"number of bands must be odd, but got {self.k}"
        self.bands = bands  # (k, d)
        # max integer small than max_block_size that divides k
        self.block_size = min(self.k, max_block_size)
        super().__init__(bands.dtype, (d_in, d_out))

    # def _rmatmat(self, v):  # v of shape (B, d_in)
    #     indices = torch.arange(self.d).view((1, self.d)).repeat((self.k, 1))  # (k, d)
    #     shifted_indices = ((indices - torch.arange(self.k)[:, None] + self.k // 2) % self.d_in)
    #     # this line can be made faster by combining multiply and sum into one
    #     muled = v[:, shifted_indices] * self.bands  # (B, k, d)
    #     muled = muled.view(v.shape[0], -1, self.d_out)
    #     result = muled.sum(-2)
    #     # assert result same as _rmatmat_block
    #     print(f'Check with block size {self.block_size}')
    #     assert torch.allclose(result, self._rmatmat_block(v), atol=1e-5), "Banded matrix multiplication is wrong"
    #     return result

    def _rmatmat(self, v):  # v of shape (B, d_in)
        B, d_in = v.shape
        k, d = self.k, self.d
        indices = torch.arange(d).view(1, d).repeat(k, 1)  # (k, d)
        shifted_indices = ((indices - torch.arange(k)[:, None] + k // 2) % d_in).long()

        result = torch.zeros(B, self.d_out, device=v.device)
        for i in range(0, k, self.block_size):
            block_indices = shifted_indices[i:i + self.block_size]  # (block_size, d)
            v_block = v[:, block_indices]  # (B, block_size, d)
            bands_block = self.bands[i:i + self.block_size]  # (block_size, d)
            muled_block = (v_block * bands_block)  # (B, block_size, d)
            # Accumulate the result
            result += muled_block.view(B, -1, self.d_out).sum(-2)
        return result


class BandedSquare(LinearOperator):
    def __init__(self, bands):
        """ LinearOperator of shape (n,n) with (n,k) matrix of bands."""
        self.k, n = bands.shape
        assert (self.k - 1) % 2 == 0, f"odd k only saw {self.k}"
        self.bands = bands.reshape(-1)
        self.bands.d_in = bands.d_in
        super().__init__(bands.dtype, (n, n))

    def _rmatmat(self, v):  # v of shape (B, n)
        n = self.shape[-1]
        indices = torch.arange(n).view((n, 1)).repeat((1, self.k))
        shifted_indices = ((indices - torch.arange(self.k) + self.k // 2) % n).view(-1)
        # this line can be made faster by combining multiply and sum into one
        muled = (v[:, shifted_indices] * self.bands).view(*v.shape[:2], self.k)
        return muled.sum(-1)

    def _matmat(self, v):
        n = self.shape[-1]
        indices = torch.arange(n).view((n, 1)).repeat((1, self.k))
        shifted_indices = (((indices - torch.arange(self.k) + self.k // 2)) % n).view(-1)
        idx = shifted_indices.reshape(n, self.k)
        bands2 = torch.gather(torch.flip(self.bands.reshape(n, self.k), (-1, )), 0, idx).reshape(-1)
        muled = (v.T[:, shifted_indices] * bands2).view(*v.T.shape[:2], self.k)
        return muled.sum(-1).T


class TeTrain(LinearOperator):
    """
    Tensor-Train operator

    Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms):
        self.Ms = Ms  # (ki, ni, mi, ki+1) ki = rank_in, ki+1 = rank_out
        dtype = self.Ms[0].dtype
        n = prod([M.shape[1] for M in Ms])
        m = prod([M.shape[2] for M in Ms])
        shape = (n, m)
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        B = v.shape[0]
        v = v.view(B, *[Mi.shape[1] for Mi in self.Ms], -1)  # (B, n1, n2, ..., nd, k1)
        for M in self.Ms:
            v = torch.einsum('bn...k,knml->b...ml', v, M)  # flop count doesn't work for tensordot
        return v.view(B, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "TT"


class BlockTeTrain(LinearOperator):
    """
    Tensor-Train operator with weights depending on the idle axes
    Implmented with einsum and einops only

    Args:
        *Ms (array_like): Cores of the TT-decomposition
    """
    def __init__(self, Ms, transpose=False):
        self.Ms = Ms  # (r_i+1, ri, m[:i], mi, ni, n[i+1:])
        dtype = self.Ms[0].dtype
        self.ms = [M.shape[2 + i] for i, M in enumerate(Ms)]
        self.ns = [M.shape[3 + i] for i, M in enumerate(Ms)]
        self.m = prod(self.ms)
        self.n = prod(self.ns)
        self.rank_out = [M.shape[0] for M in Ms]  # (r, r, ..., 1)
        self.rank_in = [M.shape[1] for M in Ms]  # (1, r, r, ..., r)
        assert self.rank_in[1:] == self.rank_out[:-1], "Rank must match"
        assert self.rank_in[0] == 1, "First rank must be 1"
        assert self.rank_out[-1] == 1, "Last rank must be 1"
        shape = (self.n, self.m)
        self.transpose = transpose
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        # v: (B, N)
        b = v.shape[0]
        for idx, M in enumerate(self.Ms):
            r = self.rank_out[idx]
            t = self.rank_in[idx]
            m = self.ms[idx]
            n = self.ns[idx]
            p = prod(self.ms[:idx])
            q = prod(self.ns[idx + 1:])
            v = v.reshape(b, t, p, n, q)
            M = M.view(r, t, p, m, n, q)
            v = torch.einsum('rtpmnq,btpnq->brpmq', M, v)
        # v: (b, 1, m1, m2, ..., m_d, 1)
        if self.transpose:
            v = v.permute(0, 1, -2, *range(v.ndim - 3, 1, -1), -1)
        return v.reshape(b, -1)

    def to_dense(self):
        raise NotImplementedError

    def __str__(self):
        return "BlockTT"


class RieBTT(LinearOperator):
    def __init__(self, Ms, shapes):
        self.Ms = Ms
        α, γ, ε, φ, ρ = self.shapes = shapes
        super().__init__(self.Ms[0].dtype, (α * γ, ε * φ))

    def _rmatmat(self, v):  # x -> x @ A
        A, B = self.Ms[0], self.Ms[1]
        return rbtt_mvm(v, A, B, self.shapes)


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

    def _rmatmat_alt(self, V):
        batch_n = V.shape[0]
        y = V.reshape(batch_n, self.ms[1], self.ms[0], -1)
        y = y.transpose(0, 1)
        y = y.reshape(self.ms[1], batch_n, self.ms[0] * self.rs[0])
        out1 = torch.bmm(y, self.Ms[0])
        out1 = out1.reshape(self.ms[1], batch_n, self.ns[0], self.rs[1])
        out1 = out1.transpose(0, 2).contiguous()
        out1 = out1.reshape(self.ns[0], batch_n, self.ms[1] * self.rs[1])
        out2 = torch.bmm(out1, self.Ms[1])
        out2 = out2.reshape(self.ns[0], batch_n, self.ns[1], self.rs[2])
        out2 = out2.transpose(0, 1)
        out2 = out2.reshape(batch_n, -1)
        return out2

    def __str__(self):
        return "BlockTT"


class GBTT(LinearOperator):
    def __init__(self, Ms, shapes):
        self.Ms = Ms
        α, _, γ, _, ε, φ, ρ, ξ = self.shapes = shapes
        super().__init__(self.Ms[0].dtype, shape=(α * γ * ξ, ε * φ * ξ))

    def _rmatmat(self, v):
        return gbtt_mvm(v, self.Ms[0], self.Ms[1], self.shapes)


class BTTDense(LinearOperator):
    def __init__(self, Ms, shapes, normalize):
        self.Ms = Ms
        self.normalize = normalize
        α, γ, ε, φ, _ = self.shapes = shapes
        super().__init__(self.Ms[0].dtype, shape=(α * γ, ε * φ))

    def _rmatmat(self, V):
        W = to_dense_btt(self.Ms[0], self.Ms[1], shapes=self.shapes)
        out = V @ W
        return out


class BTT(LinearOperator):
    def __init__(self, Ms, shapes, normalize):
        self.Ms = Ms
        self.normalize = normalize
        α, γ, ε, φ, _ = self.shapes = shapes
        super().__init__(self.Ms[0].dtype, shape=(α * γ, ε * φ))

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

    def _rmatmat(self, v):
        Ms = self.get_normalized_cores()
        return cbtt_mvm(v, Ms[0], Ms[1], self.shapes)


class Permutation(LinearOperator):
    def __init__(self, indicies, dtype):
        self.indicies = indicies
        shape = (len(self.indicies), len(self.indicies))
        super().__init__(dtype, shape)

    def _rmatmat(self, v):
        return v[:, self.indicies]
