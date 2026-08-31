"""Top-k router for Mixture-of-Experts layers."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    """Top-k expert routing with load balancing.

    Each token is routed to the top-k experts based on a learned
    gating function. Includes auxiliary losses to prevent expert
    collapse (all tokens going to the same expert).
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Learnable gating weights
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Route tokens to experts.

        Args:
            x: (num_tokens, d_model) — flattened batch*seq_len tokens

        Returns:
            top_k_scores: (num_tokens, top_k) — routing weights
            top_k_indices: (num_tokens, top_k) — which experts
            load_balance_loss: scalar — auxiliary loss for balanced routing
            metrics: dict of routing statistics
        """
        # Compute routing probabilities
        logits = self.gate(x)  # (num_tokens, num_experts)
        scores = F.softmax(logits, dim=-1)

        # Select top-k experts per token
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)

        # Renormalize top-k scores to sum to 1
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)

        # Compute auxiliary losses
        load_balance_loss = self._load_balance_loss(scores, top_k_indices)
        z_loss = self._z_loss(logits)

        # Routing metrics for monitoring
        with torch.no_grad():
            # Expert utilization: fraction of tokens routed to each expert
            expert_counts = torch.zeros(
                self.num_experts, device=x.device, dtype=torch.float
            )
            expert_counts.scatter_add_(
                0,
                top_k_indices.view(-1),
                torch.ones(top_k_indices.numel(), device=x.device),
            )
            expert_utilization = expert_counts / top_k_indices.numel()

            # Routing entropy (higher = more uniform)
            routing_entropy = -(scores * (scores + 1e-10).log()).sum(dim=-1).mean()

            metrics = {
                "routing/entropy": routing_entropy.item(),
                "routing/expert_utilization_std": expert_utilization.std().item(),
                "routing/expert_utilization_max": expert_utilization.max().item(),
                "routing/expert_utilization_min": expert_utilization.min().item(),
                "routing/load_balance_loss": load_balance_loss.item(),
                "routing/z_loss": z_loss.item(),
            }

        return top_k_scores, top_k_indices, load_balance_loss + 0.1 * z_loss, metrics

    def _load_balance_loss(
        self, scores: torch.Tensor, top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """Switch Transformer load balancing loss.

        Encourages each expert to receive an equal fraction of tokens.

        L_balance = N * sum_i(f_i * p_i)
        where:
            f_i = fraction of tokens routed to expert i
            p_i = mean routing probability for expert i
        """
        num_tokens = scores.size(0)

        # f_i: fraction of tokens dispatched to each expert
        expert_mask = F.one_hot(top_k_indices, self.num_experts).sum(dim=1)  # (tokens, experts)
        f = expert_mask.float().mean(dim=0)  # (experts,)

        # p_i: mean routing probability for each expert
        p = scores.mean(dim=0)  # (experts,)

        return self.num_experts * (f * p).sum()

    def _z_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """Z-loss: regularizes router logits to prevent instability.

        From ST-MoE paper. Prevents router logits from growing too large.
        """
        return logits.float().square().mean()
