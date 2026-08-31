"""MoE transformer — sparse Mixture-of-Experts variant."""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .moe_layer import MoELayer


class MoETransformerBlock(nn.Module):
    """Transformer block with MoE FFN replacing dense FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: int | None = None,
        capacity_factor: float = 1.25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.RMSNorm(d_model)
        self.moe = MoELayer(d_model, num_experts, top_k, d_ff, capacity_factor, dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            x: (batch, seq_len, d_model)
            aux_loss: scalar load balancing loss from this layer
            metrics: routing metrics dict
        """
        x = x + self.attn(self.ln1(x), mask)
        moe_out, aux_loss, metrics = self.moe(self.ln2(x))
        x = x + moe_out
        return x, aux_loss, metrics


class MoETransformer(nn.Module):
    """Mixture-of-Experts decoder-only transformer.

    Same architecture as DenseTransformer, but every FFN layer is replaced
    with an MoE layer containing N experts, of which top-k are activated
    per token. This gives more total parameters but similar active FLOPs.

    Config philosophy:
        - Dense baseline: 85M params, all active
        - MoE-8E Top-2: ~250M total, ~85M active per token
        → Same compute budget, but the model can specialize experts
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        num_experts: int = 8,
        top_k: int = 2,
        d_ff: int | None = None,
        capacity_factor: float = 1.25,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        load_balance_coeff: float = 0.01,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.load_balance_coeff = load_balance_coeff

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MoETransformerBlock(
                d_model, n_heads, num_experts, top_k, d_ff, capacity_factor, dropout
            )
            for _ in range(n_layers)
        ])

        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len))
        self.register_buffer("causal_mask", mask)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> dict:
        """
        Args:
            input_ids: (batch, seq_len)
            targets: (batch, seq_len) optional targets for loss

        Returns:
            dict with keys:
                logits: (batch, seq_len, vocab_size)
                loss: scalar (if targets provided) — LM loss + load balance loss
                lm_loss: scalar LM loss only
                aux_loss: scalar total auxiliary loss
                routing_metrics: list of per-layer routing metrics
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len

        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        total_aux_loss = 0.0
        all_routing_metrics = []

        for block in self.blocks:
            x, aux_loss, metrics = block(x, self.causal_mask)
            total_aux_loss = total_aux_loss + aux_loss
            all_routing_metrics.append(metrics)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        result = {
            "logits": logits,
            "aux_loss": total_aux_loss,
            "routing_metrics": all_routing_metrics,
        }

        if targets is not None:
            lm_loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            result["lm_loss"] = lm_loss
            result["loss"] = lm_loss + self.load_balance_coeff * total_aux_loss

        return result

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        # Approximate active params: shared params + 1 expert per MoE layer * top_k
        # This is a rough estimate
        shared = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "experts" not in name
        )
        expert_per_layer = sum(
            p.numel()
            for p in self.blocks[0].moe.experts.experts[0].parameters()
        )
        n_layers = len(self.blocks)
        top_k = self.blocks[0].moe.router.top_k
        active = shared + expert_per_layer * top_k * n_layers

        return {
            "total": total,
            "active_per_token": active,
            "total_million": total / 1e6,
            "active_million": active / 1e6,
        }
