"""Learning rate schedule: linear warmup + cosine decay."""

import math


def get_lr(
    step: int,
    max_lr: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Compute learning rate with linear warmup and cosine decay.

    Args:
        step: Current training step
        max_lr: Peak learning rate (reached after warmup)
        min_lr: Minimum learning rate (at end of cosine decay)
        warmup_steps: Number of linear warmup steps
        max_steps: Total training steps
    """
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step + 1) / warmup_steps

    if step >= max_steps:
        return min_lr

    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
