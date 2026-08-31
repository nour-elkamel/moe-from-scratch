"""Dense transformer baseline — standard decoder-only GPT-style model."""

import math
import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .ffn import FeedForward


class TransformerBlock(nn.Module):
    """Standard transformer block: attention + FFN with pre-norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class DenseTransformer(nn.Module):
    """Dense decoder-only transformer for language modeling.

    This is the baseline model to compare against the MoE version.
    Both have the same number of active parameters per forward pass.
    """

    def __init__(
        self,
        vocab_size: int = 50304,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int | None = None,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        # Causal mask (registered buffer, not a parameter)
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
            input_ids: (batch, seq_len) token indices
            targets: (batch, seq_len) target token indices for loss

        Returns:
            dict with keys: logits, loss (if targets provided)
        """
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        pos = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_emb(input_ids) + self.pos_emb(pos))

        for block in self.blocks:
            x = block(x, self.causal_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        result = {"logits": logits}

        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
