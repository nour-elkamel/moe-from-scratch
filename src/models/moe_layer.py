"""MoE layer: combines router + experts to replace a standard FFN."""

import torch
import torch.nn as nn

from .router import TopKRouter
from .experts import ExpertCollection


class MoELayer(nn.Module):
    """Mixture-of-Experts layer.

    Replaces a standard FFN in a transformer block. Each token is routed
    to the top-k experts by the router, processed by those experts,
    and the outputs are combined via weighted sum.

    Architecture:
        input -> Router -> dispatch to top-k Experts -> weighted combine -> output
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: int | None = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.router = TopKRouter(d_model, num_experts, top_k, capacity_factor)
        self.experts = ExpertCollection(num_experts, d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
            aux_loss: scalar load balancing loss
            metrics: routing statistics dict
        """
        batch, seq_len, d_model = x.shape

        # Flatten to (num_tokens, d_model) for routing
        x_flat = x.view(-1, d_model)

        # Route tokens to experts
        top_k_scores, top_k_indices, aux_loss, metrics = self.router(x_flat)

        # Dispatch and combine expert outputs
        output_flat = self.experts(x_flat, top_k_scores, top_k_indices)

        # Reshape back
        output = output_flat.view(batch, seq_len, d_model)

        return output, aux_loss, metrics
