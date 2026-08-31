"""Loss functions for language modeling."""

import torch
import torch.nn.functional as F


def compute_lm_loss(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -1
) -> torch.Tensor:
    """Standard next-token prediction loss."""
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
