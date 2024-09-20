import torch


class BlockTeTr(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def forward(ctx, x, A, B, shapes, gate=None, num_active=None):
        # x(γ, α) -> y(φ, ε): α -> φ, γ -> ε
        # A (γ, α, φ ρ)
        # B (φ, γ ρ, ε)
        bsz = x.shape[0]
        (_, ρ, _), (α, γ), (φ, ε) = rs, ms, ns = shapes
        assert rs[0] == rs[2] == 1, 'BTT first and last rank should be 1'
        Z = torch.empty(bsz, γ, φ * ρ, device=x.device, dtype=x.dtype).transpose(0, 1)
        Y = torch.empty(bsz, φ, ε, device=x.device, dtype=x.dtype).transpose(0, 1)
        ctx.shapes = shapes

        X = x.reshape(bsz, γ, α)  # (bsz, γ, α)
        X = X.transpose(0, 1)  # (γ, bsz, α)
        X = X.reshape(γ, bsz, α)
        torch.bmm(X, A, out=Z)  # (γ, bsz, φ ρ)
        Z = Z.reshape(γ, bsz, φ, ρ)  # (γ, bsz, φ, ρ)
        if gate is not None:
            # (bsz, ρ) -> (γ, bsz, φ, ρ)
            Z = Z * gate[None, :, None, :]
        Z = Z.transpose(0, 2).contiguous()  # (φ, bsz, γ, ρ)
        Z = Z.reshape(φ, bsz, γ * ρ)  # (φ, bsz, γ ρ)
        torch.bmm(Z, B, out=Y)  # (φ, bsz, ε)
        Y = Y.reshape(φ, bsz, ε)  # (φ, bsz, ε)
        Y = Y.transpose(0, 1)  # (bsz, φ, ε)
        Y = Y.reshape(bsz, -1)  # (bsz, φ ε)
        ctx.save_for_backward(X, A, B, Z, gate)
        return Y

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        X, A, B, z, gate = ctx.saved_tensors
        bsz = X.shape[1]
        (_, ρ, _), (α, γ), (φ, ε) = rs, ms, ns = ctx.shapes

        dZ = torch.empty(bsz, φ, ρ * γ, device=X.device, dtype=X.dtype).transpose(1, 0)
        dx = torch.empty(bsz, γ, α, device=X.device, dtype=X.dtype).transpose(1, 0)

        dY = grad.reshape(bsz, φ, ε).transpose(1, 0)
        dB = torch.bmm(dY.transpose(1, 2), z).transpose(1, 2)
        torch.bmm(dY, B.transpose(1, 2), out=dZ)  # dZ (φ, bsz, γ ρ)
        dZ = dZ.reshape(φ, bsz, γ, ρ)

        if gate is not None:
            # for sparse MoE, when gate is 0, dgate is 0 due to taking topk
            gate_inv = torch.where(gate == 0, torch.zeros_like(gate), 1 / gate)
            # z: (φ, bsz, γ ρ), dz: (φ, bsz, γ ρ), gate: (bsz, ρ)
            z_ungated = z.reshape(φ, bsz, γ, ρ) * gate_inv[None, :, None, :]
            dgate = (dZ * z_ungated).sum(dim=(0, 2))  # (bsz, ρ)
            # Multiply dz by the gate vector
            dZ = dZ * gate[None, :, None, :]
        else:
            dgate = None

        dZ = dZ.transpose(0, 2).contiguous().reshape(γ, bsz, φ * ρ)

        torch.bmm(dZ, A.transpose(1, 2), out=dx)
        dx = dx.reshape(γ, bsz, α)
        dx = dx.transpose(1, 0).reshape(bsz, -1)

        dA = torch.bmm(dZ.transpose(1, 2), X).transpose(1, 2)
        return dx, dA, dB, None, dgate, None

    @staticmethod
    @torch.cuda.amp.custom_fwd(
        cast_inputs=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16)
    def _pull_back(grad, W1, W2, shapes):
        B = grad.shape[0]
        rs, ms, ns = shapes
        grad_re = grad.reshape(B, ns[0], ns[1]).transpose(1, 0)
        aux = torch.empty(B, ns[0], rs[1] * ms[1], device=grad.device, dtype=grad.dtype).transpose(1, 0)
        dx = torch.empty(B, ms[1], ms[0], device=grad.device, dtype=grad.dtype).transpose(1, 0)

        torch.bmm(grad_re, W2.transpose(1, 2), out=aux)
        aux = aux.reshape(ns[0], B, ms[1], rs[1]).transpose(0, 2).contiguous()
        aux = aux.reshape(ms[1], B, ns[0] * rs[1])
        torch.bmm(aux, W1.transpose(1, 2), out=dx)
        dx = dx.reshape(ms[1], B, ms[0] * rs[0])
        dx = dx.transpose(1, 0).reshape(B, -1)
        return dx


btt_mvm = BlockTeTr.apply
btt_mvmt = BlockTeTr._pull_back
