import torch


class BlockTeTr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W1, W2, shapes):
        rs, ms, ns = shapes
        batch_n = x.shape[0]
        ctx.shapes = shapes

        # y = x.reshape(batch_n, ms[1], ms[0], -1)
        # y = y.transpose(0, 1)
        y = x.reshape(batch_n, ms[0], ms[1], -1)
        y = y.permute(2, 1, 0, 3)
        y = y.reshape(ms[1], batch_n, ms[0] * rs[0])
        out1 = torch.bmm(y, W1)
        out1 = out1.reshape(ms[1], batch_n, ns[0], rs[1])
        out1 = out1.transpose(0, 2)
        out1 = out1.reshape(ns[0], batch_n, ms[1] * rs[1])
        out2 = torch.bmm(out1, W2)
        out2 = out2.reshape(ns[0], batch_n, ns[1], rs[2])
        out2 = out2.transpose(0, 1)
        out2 = out2.reshape(batch_n, -1)
        ctx.save_for_backward(x, W1, W2, out1)
        # return out2
        return out2.clone()

    @staticmethod
    def backward(ctx, grad):
        x, W1, W2, out1 = ctx.saved_tensors
        B = x.shape[0]
        rs, ms, ns = ctx.shapes
        grad_re = grad.reshape(B, ns[0], ns[1]).transpose(1, 0)

        aux = torch.bmm(grad_re, W2.transpose(1, 2))
        aux = aux.reshape(ns[0], B, ms[1], rs[1]).transpose(0, 2)
        aux = aux.reshape(ms[1], B, ns[0] * rs[1])
        dx = torch.bmm(aux, W1.transpose(1, 2))
        dx = dx.reshape(ms[1], B, ms[0] * rs[0])
        dx = dx.transpose(1, 0).reshape(B, -1)

        x_res = x.reshape(B, ms[1], ms[0]).transpose(0, 1)
        dW1 = torch.bmm(aux.transpose(1, 2), x_res).transpose(1, 2)

        dW2 = torch.bmm(grad_re.transpose(1, 2), out1).transpose(1, 2)
        return dx, dW1, dW2, None


btt_mvm = BlockTeTr.apply


def prep_cores(cores):
    rs, ms, ns = infer_shapes(cores)
    if len(ms) == 2:
        core1 = cores[0].permute(3, 1, 0, 2, 4)
        core1 = core1.reshape(ms[1], ms[0] * rs[0], ns[0] * rs[1])
        core2 = cores[1].permute(3, 1, 0, 2, 4)
        core2 = core2.reshape(ns[0], ms[1] * rs[1], ns[1] * rs[2])
        return core1, core2
    else:
        raise NotImplementedError


def infer_shapes(cores):
    rs = [M.shape[0] for M in cores] + [cores[-1].shape[-1]]
    ms = [M.shape[1] for M in cores]
    ns = [M.shape[2] for M in cores]
    return rs, ms, ns
