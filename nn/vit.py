# adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .cola_nn import dense_init

# helpers


class CappedList():
    def __init__(self, max_len=1):
        self.max_len = max_len
        self.buffer = []

    def append(self, x):
        if len(self.buffer) < self.max_len:
            self.buffer.append(x.cpu())


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MoELayer(nn.Module):
    def __init__(self, experts, in_features, out_features, k=2):
        super().__init__()
        assert len(experts) > 0
        self.in_features = in_features
        self.out_features = out_features
        self.experts = nn.ModuleList(experts)
        self.k = min(k, len(experts))
        self.moe_gate = nn.Linear(in_features, len(experts), bias=False)
        # zero init gate to make initial selection uniform and obey µP
        self.moe_gate.weight.data.zero_()

    def load_balancing_loss_fn(self, gate_logits):
        num_experts = gate_logits.shape[1]
        gate_probs = F.softmax(gate_logits, dim=1)  # (B, E)
        probs = gate_probs.mean(dim=0)  # (E)
        max_idx = gate_probs.argmax(dim=1)  # (B)
        expert_mask = F.one_hot(max_idx, num_classes=num_experts)  # (B, E)
        fracs = expert_mask.float().mean(dim=0)  # (E)
        loss = (fracs * probs).sum() * num_experts
        return loss

    def forward(self, inputs):
        batch_shape = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.moe_gate(inputs)
        self.moe_gate.load_balancing_loss = self.load_balancing_loss_fn(gate_logits)
        if not self.training:
            # compute perplexity of gate logits
            gate_probs = F.softmax(gate_logits, dim=1)
            agg_gate_probs = gate_probs.mean(dim=0)
            ppl = torch.exp(-torch.sum(gate_probs * torch.log(gate_probs), dim=1))
            agg_ppl = torch.exp(-torch.sum(agg_gate_probs * torch.log(agg_gate_probs), dim=0))
            self.moe_gate.weight.ppl = ppl.mean().item()  # should be low
            self.moe_gate.weight.agg_ppl = agg_ppl.item()  # should be high

        weights, selected_experts = torch.topk(gate_logits, self.k, dim=-1)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)

        results = torch.zeros(inputs.shape[0], self.out_features, device=inputs.device)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            if len(batch_idx) == 0:
                continue
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(inputs[batch_idx])
        return results.view(*batch_shape, -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., fixup=False, use_bias=True):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim, bias=use_bias), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden_dim, dim, bias=use_bias), nn.Dropout(dropout))

        # a scaler multiplier
        self.out_scalar = nn.Parameter(torch.ones(1)) if fixup else None

        # zero init
        dense_init(self.net[-2], zero_init=True)

    def forward(self, x):
        out = self.net(x)
        if self.out_scalar is not None:
            out = out * self.out_scalar
        return out


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., fixup=False, attn_mult=1, use_bias=True, causal=False):
        super().__init__()
        self.causal = causal
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = attn_mult * 8 / dim_head  # μP prescribs this scaling, 8 for backward compatibility at dim_head=64

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # split into 3 separate linears so that CoLA replaces each individually
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.q_norm = nn.LayerNorm(dim_head)
        self.k_norm = nn.LayerNorm(dim_head)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim, bias=use_bias),
                                    nn.Dropout(dropout)) if project_out else nn.Identity()

        # a scaler multiplier
        self.out_scalar = nn.Parameter(torch.ones(1)) if fixup else None

        # zero init W_Q per μP recommendation
        dense_init(self.to_q, zero_init=True)
        # zero init
        dense_init(self.to_out[0], zero_init=True)

    def forward(self, x):
        x = self.norm(x)

        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = self.q_norm(q)
        k = self.k_norm(k)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # b h n n

        # attention mask
        if self.causal:
            mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1)
            dots = dots.masked_fill(mask == 1, float('-inf'))

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.out_scalar is not None:
            out = out * self.out_scalar
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., fixup=False, attn_mult=1, num_ffn_experts=1,
                 use_bias=True, causal=False, last_layernorm=True):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if last_layernorm else nn.Identity()
        self.layers = nn.ModuleList([])
        self.fixup = fixup
        self.attn_mult = attn_mult
        for _ in range(depth):
            if num_ffn_experts > 1:
                ffn = MoELayer(
                    [FeedForward(dim, mlp_dim, dropout=dropout, fixup=fixup, use_bias=use_bias) for _ in range(num_ffn_experts)],
                    dim, dim)
            else:
                ffn = FeedForward(dim, mlp_dim, dropout=dropout, fixup=fixup, use_bias=use_bias)
            attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, fixup=fixup, attn_mult=attn_mult,
                             use_bias=use_bias, causal=causal)
            self.layers.append(nn.ModuleList([attn, ffn]))
        self.hs = [CappedList() for _ in range(depth + 2)]

    def forward(self, x):
        if not self.training:
            self.hs[0].append(x.detach())
        for i, (attn, ff) in enumerate(self.layers):
            x = attn(x) + x
            x = ff(x) + x
            if not self.training:
                self.hs[i + 1].append(x.detach())
        return self.norm(x)


class ViT(nn.Module):
    def __init__(self, dim_out, width, depth, ffn_expansion=4, heads=8, dim_head=None, image_size=32, patch_size=8, pool='cls',
                 in_channels=3, dropout=0., fixup=False, attn_mult=1, output_mult=1, emb_mult=1, use_bias=True, num_ffn_experts=1,
                 **kwargs):
        super().__init__()
        self.fixup = fixup
        self.emb_mult = emb_mult
        self.attn_mult = attn_mult
        self.output_mult = output_mult
        if dim_head is None:
            dim_head = width / heads
            assert int(dim_head) == dim_head, 'dimension of each head must be integer'
            dim_head = int(dim_head)
            print(f"Setting dim_head to {dim_head}")
        mlp_dim = ffn_expansion * width
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Invalid patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = in_channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, width, bias=use_bias),
            nn.LayerNorm(width),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, width))
        self.pos_embedding.fan_in_dims = (1, )  # constant fan in
        self.pos_embedding.fan_out_dims = (-1, )
        self.cls_token = nn.Parameter(torch.randn(1, 1, width))
        self.cls_token.fan_in_dims = (1, )  # constant fan in
        self.cls_token.fan_out_dims = (-1, )
        self.dropout = nn.Dropout(dropout)

        self.transformer = Transformer(width, depth, heads, dim_head, mlp_dim, dropout, fixup, attn_mult, num_ffn_experts,
                                       use_bias)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(width, dim_out, bias=use_bias)

        if not use_bias:
            # Freeze all emb and pos embeddings
            emb_params = [self.pos_embedding, self.cls_token] + list(self.to_patch_embedding.parameters())
            for p in emb_params:
                p.requires_grad = False

        # fix init
        self.fix_init()
        # logs
        self.hs = [CappedList() for _ in range(depth + 2)]

    def fix_init(self):
        # go through all linear layers
        for m in self.modules():
            # skip zero init layers
            if isinstance(m, nn.Linear):
                if hasattr(m.weight, 'zero_init') and m.weight.zero_init:
                    print(f"Skipping zero init: {m}")
                    continue
                else:
                    print(f"Fixing init: {m}")
                    dense_init(m)
        print(f"Fixing output init: {self.mlp_head}")
        dense_init(self.mlp_head, zero_init=True)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = x * self.emb_mult
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        y = self.mlp_head(x) * self.output_mult
        if not self.training:
            self.transformer.hs[-1].append(y.detach())
        return y

    def get_features(self):
        return self.transformer.hs

    def clear_features(self):
        self.transformer.hs = [CappedList() for _ in range(len(self.hs))]


class ARTransformer(nn.Module):
    def __init__(self, dim_out, width, depth, ffn_expansion=4, heads=8, dim_head=None, image_size=32, in_channels=3,
                 vocab_size=256, num_ffn_experts=1, last_layernorm=True, **kwargs):
        super().__init__()
        if dim_head is None:
            dim_head = width / heads
            assert int(dim_head) == dim_head, 'dimension of each head must be integer'
            dim_head = int(dim_head)
            print(f"Setting dim_head to {dim_head}")
        mlp_dim = ffn_expansion * width

        seq_len = image_size * image_size * in_channels

        self.embedding = nn.Embedding(vocab_size, width)
        self.embedding.weight.fan_in_dims = (0, )  # constant fan in
        self.embedding.weight.fan_out_dims = (0, )

        self.bos_token = nn.Parameter(torch.randn(1, 1, width))
        self.bos_token.fan_in_dims = (1, )  # constant fan in
        self.bos_token.fan_out_dims = (-1, )

        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, width))
        self.pos_embedding.fan_in_dims = (1, )  # constant fan in

        self.embedding.weight.data *= 0.02
        self.pos_embedding.data *= 0.02
        self.pos_embedding.fan_out_dims = (-1, )
        self.bos_token.data *= 0.02

        self.transformer = Transformer(width, depth, heads, dim_head, mlp_dim, dropout=0., fixup=False, attn_mult=1,
                                       num_ffn_experts=1, use_bias=True, causal=True, last_layernorm=last_layernorm)

        self.mlp_head = nn.Linear(width, vocab_size)

        # fix init
        self.fix_init()
        # logs
        self.hs = [CappedList() for _ in range(depth + 2)]

    def fix_init(self):
        # go through all linear layers
        for m in self.modules():
            # skip zero init layers
            if isinstance(m, nn.Linear):
                if hasattr(m.weight, 'zero_init') and m.weight.zero_init:
                    print(f"Skipping zero init: {m}")
                    continue
                else:
                    print(f"Fixing init: {m}")
                    dense_init(m)
        print(f"Fixing output init: {self.mlp_head}")
        dense_init(self.mlp_head, zero_init=True)

    def forward(self, x):
        x = self.embedding(x)  # b c h w d
        # b c h w d -> b (h w c) d
        x = rearrange(x, 'b c h w d -> b (h w c) d')
        b, n, _ = x.shape

        # add bos token
        bos_tokens = repeat(self.bos_token, '() n d -> b n d', b=b)
        x = torch.cat((bos_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.transformer(x)

        y = self.mlp_head(x)  # (b, n, vocab_size)
        if not self.training:
            self.transformer.hs[-1].append(y.detach())
        return y

    def get_features(self):
        return self.transformer.hs

    def clear_features(self):
        self.transformer.hs = [CappedList() for _ in range(len(self.hs))]
