"""Perplexity evaluation for language models."""

import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate_perplexity(model, dataloader: DataLoader, device: str = "cuda") -> float:
    """Compute perplexity on a dataset.

    For MoE models, only uses the LM loss (not auxiliary loss) for perplexity.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        output = model(input_ids, targets=targets)
        lm_loss = output.get("lm_loss", output["loss"])

        num_tokens = (targets != -1).sum().item()
        total_loss += lm_loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity
