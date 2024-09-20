import torch
from math import prod
from functools import partial
import cola
from cola.ops.operators import Jacobian
from learning.fns import to_dense_btt_flat


class RieBTTmvm(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, x, A, B, shapes):
        # A: (γ, α, φ ρ)
        # B: (φ, γ ρ, ε)
        α, γ, ε, φ, ρ = shapes
        bsz = x.shape[0]

        X = x.reshape(bsz, α, γ)
        X = X.transpose(0, 2).transpose(1, 2)  # (γ, bsz, α)
        Z = torch.bmm(X, A)  # (γ, bsz, φ ρ)
        Z = Z.reshape(γ, bsz, φ, ρ)
        Z = Z.transpose(0, 2).contiguous()  # (φ, bsz, γ, ρ)
        Z = Z.reshape(φ, bsz, γ * ρ)
        Y = torch.bmm(Z, B)  # (φ, bsz, ε)
        Y = Y.transpose(0, 1).transpose(1, 2)  # (bsz, ε, φ)
        Y = Y.reshape(bsz, -1)

        ctx.shapes = shapes
        ctx.save_for_backward(X, A, B, Z)
        return Y

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        X, A, B, Z = ctx.saved_tensors
        bsz = X.shape[1]
        α, γ, ε, φ, ρ = ctx.shapes

        grad_re = grad.reshape(bsz, ε, φ)
        grad_re = grad_re.transpose(0, 2).transpose(1, 2)  # (φ, bsz, ε)
        aux = torch.bmm(grad_re, B.transpose(1, 2))  # (φ, bsz, γ ρ)
        aux = aux.reshape(φ, bsz, γ, ρ)
        aux = aux.transpose(0, 2).contiguous()  # (γ, bsz, φ, ρ)
        aux = aux.reshape(γ, bsz, φ * ρ)

        # TODO: Should I save compute from not running dx?
        # dx = torch.bmm(aux, A.transpose(1, 2))  # (γ, bsz, α)
        # dx = dx.transpose(0, 1).transpose(1, 2).reshape(bsz, -1)
        dA = torch.bmm(aux.transpose(1, 2), X).transpose(1, 2)  # (γ, α, φ ρ)
        dB = torch.bmm(grad_re.transpose(1, 2), Z).transpose(1, 2)  # (φ, γ ρ, ε)

        F = torch.concat((A.reshape(-1), B.reshape(-1)))
        fn = partial(to_dense_btt_flat, shapes=ctx.shapes)
        J = Jacobian(fn, F)
        JTJ = J.T @ J
        JTJ = cola.PSD(JTJ)
        dF = torch.concat((dA.reshape(-1), dB.reshape(-1)))
        JTJ_inv = cola.linalg.inv(JTJ, alg=cola.CG(max_iters=1_000, tol=6e-6)) @ dF
        JTJ_dF0, JTJ_dF1 = JTJ_inv[:prod(A.shape)], JTJ_inv[prod(A.shape):]
        JTJ_dF0, JTJ_dF1 = JTJ_dF0.reshape(A.shape), JTJ_dF1.reshape(B.shape)

        return None, JTJ_dF0, JTJ_dF1, None


rbtt_mvm = RieBTTmvm.apply
