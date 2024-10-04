import torch
from torch import nn
from torch.nn import functional as F

class MoELayer(nn.Module):
    def __init__(self, experts, in_features, out_features, k=2):
        super().__init__()
        assert len(experts) > 0
        self.in_features = in_features
        self.out_features = out_features
        self.experts = nn.ModuleList(experts)
        self.num_experts = len(experts)
        self.num_active_experts = min(k, len(experts))
        self.moe_gate = nn.Linear(in_features, len(experts), bias=False)
        self.moe_gate.leave_as_dense = True
        # zero init gate to make initial selection uniform and obey µP
        self.moe_gate.weight.data.zero_()

    def load_balancing_loss_fn(self, gate_logits, top_k_weights, selected_experts):
        probs = F.softmax(gate_logits, dim=1)  # (B, E)
        zeros = torch.zeros_like(probs) # (B, E)
        zeros = zeros.to(top_k_weights.dtype) 
        gates = zeros.scatter(1, selected_experts, top_k_weights) # (B, E)
        return self.compute_aux_loss(probs, gate_logits, gates)

    def compute_aux_loss(self, probs, logits, gates):
        """
        Calculate and return the auxiliary loss based on the accumulated statistics.

        Args:
            eps (float): Small epsilon value for numerical stability.

        Returns:
            torch.Tensor: The calculated auxiliary loss.
        """
        count = logits.size(0)
        probs = probs.sum(0) # unnormalized marginal expert probs
        freq = (gates > 0).float().sum(0) # unnoramlized marginal freqs
        lsesq = (torch.log(torch.exp(logits).sum(dim=-1)) ** 2).sum() # squared log partition functions

        switchloss = self.num_experts * (F.normalize(probs, p=1, dim=0) * F.normalize(freq, p=1, dim=0)).sum()
        zloss = lsesq / count
        loss = switchloss + 0.1 * zloss
        return loss
    
    def forward(self, inputs):
        batch_shape = inputs.shape[:-1]
        inputs = inputs.view(-1, inputs.shape[-1])
        gate_logits = self.moe_gate(inputs)  # + 1e-6 * torch.randn(inputs.shape[0], len(self.experts), device=inputs.device)
        top_k_weights, selected_experts = torch.topk(gate_logits, self.num_active_experts, dim=-1)  # (B, k), (B, k)
        top_k_weights = F.softmax(top_k_weights, dim=1, dtype=torch.float).to(inputs.dtype)  # (B, k)
        self.moe_gate.load_balancing_loss = self.load_balancing_loss_fn(gate_logits, top_k_weights, selected_experts)
        if not self.training:
            # compute perplexity of gate logits
            gate_probs = F.softmax(gate_logits, dim=1)
            agg_gate_probs = gate_probs.mean(dim=0)
            ppl = torch.exp(-torch.sum(gate_probs * torch.log(gate_probs), dim=1))
            agg_ppl = torch.exp(-torch.sum(agg_gate_probs * torch.log(agg_gate_probs), dim=0))
            self.moe_gate.weight.ppl = ppl.mean().item()  # should be low
            self.moe_gate.weight.agg_ppl = agg_ppl.item()  # should be high

        # Compute indices and inputs for each expert
        active_expert_indices = torch.unique(selected_experts)
        active_experts = [self.experts[i] for i in active_expert_indices]
        batch_idxs = [torch.where(selected_experts == i)[0] for i in active_expert_indices]
        nth_experts = [torch.where(selected_experts == i)[1] for i in active_expert_indices]
        expert_inputs = [inputs[batch_idx] for batch_idx in batch_idxs]

        expert_outputs = []
        for expert_in, expert in zip(expert_inputs, active_experts):
            expert_outputs.append(expert(expert_in))

        results = torch.zeros(inputs.shape[0], self.out_features, device=inputs.device)
        for expert_output, batch_idx, nth_expert in zip(expert_outputs, batch_idxs, nth_experts):
            results[batch_idx] += top_k_weights[batch_idx, nth_expert, None] * expert_output
        return results.view(*batch_shape, -1)