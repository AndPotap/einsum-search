import torch


def non_btt_mvm(x, F0, F1, shapes, act_fn):
    rs, ms, ns = shapes
    batch_n = x.shape[0]

    y = x.reshape(batch_n, ms[1], ms[0], -1)
    y = y.transpose(0, 1)
    y = y.reshape(ms[1], batch_n, ms[0] * rs[0])
    out1 = torch.bmm(y, F0)
    out1 = act_fn(out1)
    out1 = out1.reshape(ms[1], batch_n, ns[1], rs[1])
    out1 = out1.transpose(0, 2).contiguous()
    out1 = out1.reshape(ns[1], batch_n, ms[1] * rs[1])
    out2 = torch.bmm(out1, F1)
    out2 = out2.reshape(ns[1], batch_n, ns[0], rs[2])
    # out2 = act_fn(out2)
    out2 = out2.transpose(0, 1)
    out2 = out2.transpose(1, 2)
    out2 = out2.reshape(batch_n, -1)
    return out2
