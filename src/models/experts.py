"""Expert collection — the pool of FFN experts in an MoE layer."""

import torch
import torch.nn as nn

from .ffn import FeedForward


class ExpertCollection(nn.Module):
    """A collection of N expert FFN networks.

    Each expert is a standard FeedForward network. During forward pass,
    tokens are dispatched to their assigned experts based on routing decisions.
    """

    def __init__(
        self,
        num_experts: int,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout) for _ in range(num_experts)]
        )

    def forward(
        self,
        x: torch.Tensor,
        top_k_scores: torch.Tensor,
        top_k_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch tokens to experts and combine outputs.

        Args:
            x: (num_tokens, d_model)
            top_k_scores: (num_tokens, top_k) — routing weights
            top_k_indices: (num_tokens, top_k) — expert assignments

        Returns:
            output: (num_tokens, d_model) — weighted combination of expert outputs
        """
        num_tokens, d_model = x.shape
        top_k = top_k_indices.size(1)

        # Initialize output
        output = torch.zeros_like(x)

        # Process each expert
        for expert_idx in range(self.num_experts):
            # Find which (token, k) pairs are routed to this expert
            mask = (top_k_indices == expert_idx)  # (num_tokens, top_k)

            if not mask.any():
                continue

            # Get token indices routed to this expert
            token_indices = mask.any(dim=1).nonzero(as_tuple=True)[0]

            if len(token_indices) == 0:
                continue

            # Run expert on selected tokens
            expert_input = x[token_indices]  # (n_selected, d_model)
            expert_output = self.experts[expert_idx](expert_input)

            # Get routing weights for these tokens to this expert
            # For each selected token, find the weight assigned to this expert
            weights = torch.zeros(len(token_indices), device=x.device)
            for k in range(top_k):
                k_mask = top_k_indices[token_indices, k] == expert_idx
                weights[k_mask] = top_k_scores[token_indices[k_mask], k]

            # Weighted accumulation
            output[token_indices] += weights.unsqueeze(-1) * expert_output

        return output
