import torch


class BTTGen(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, x, A, B, shapes):
        # A: (γ ξ, α, φ ρ)
        # B: (φ ξ, γ ρ, ε)
        α, _, γ, _, ε, φ, ρ, ξ = shapes
        bsz = x.shape[0]

        X = x.reshape(bsz, α, γ, ξ)
        X = X.transpose(0, 2).transpose(1, 3)  # (γ, ξ, bsz, α)
        X = X.reshape((γ * ξ), bsz, α)
        Z = torch.bmm(X, A)  # (γ ξ, bsz, φ ρ)
        Z = Z.reshape(γ, ξ, bsz, φ, ρ)
        Z = Z.transpose(0, 3).contiguous()  # (φ, ξ, bsz, γ, ρ)
        Z = Z.reshape(φ * ξ, bsz, γ * ρ)
        Y = torch.bmm(Z, B)
        Y = Y.reshape(φ, ξ, bsz, ε)
        Y = Y.transpose(0, 2).transpose(1, 3)  # (bsz, ε, φ, ξ)
        Y = Y.reshape(bsz, -1)

        ctx.shapes = shapes
        ctx.save_for_backward(X, A, B, Z)
        return Y

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        X, A, B, Z = ctx.saved_tensors
        bsz = X.shape[1]
        α, _, γ, _, ε, φ, ρ, ξ = ctx.shapes

        grad_re = grad.reshape(bsz, ε, φ, ξ)
        grad_re = grad_re.transpose(1, 3).transpose(0, 2)  # (φ, ξ, bsz, ε)
        grad_re = grad_re.reshape(φ * ξ, bsz, ε)
        aux = torch.bmm(grad_re, B.transpose(1, 2))  # (φ ξ, bsz, γ ρ)
        aux = aux.reshape(φ, ξ, bsz, γ, ρ)
        aux = aux.transpose(0, 3).contiguous()  # (γ, ξ, bsz, φ, ρ)
        aux = aux.reshape(γ * ξ, bsz, φ * ρ)

        # TODO: Should I save compute from not running dx?
        dx = torch.bmm(aux, A.transpose(1, 2))  # (γ ξ, bsz, α)
        dx = dx.reshape(γ, ξ, bsz, α)
        dx = dx.transpose(1, 3).transpose(0, 2).reshape(bsz, -1)

        dA = torch.bmm(aux.transpose(1, 2), X).transpose(1, 2)  # (γ ξ, α, φ ρ)

        dB = torch.bmm(grad_re.transpose(1, 2), Z).transpose(1, 2)  # (φ ξ, γ ρ, ε)
        return dx, dA, dB, None


gbtt_mvm = BTTGen.apply
