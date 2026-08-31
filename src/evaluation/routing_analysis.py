"""Routing analysis and visualization for MoE models."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader


@torch.no_grad()
def analyze_routing(
    model, dataloader: DataLoader, device: str = "cuda", max_batches: int = 50
) -> dict:
    """Analyze expert routing patterns across the model.

    Collects per-layer routing statistics and returns them for visualization.
    """
    model.eval()

    n_layers = len(model.blocks)
    n_experts = model.blocks[0].moe.router.num_experts

    # Accumulators
    expert_counts = torch.zeros(n_layers, n_experts)
    routing_entropies = torch.zeros(n_layers)
    n_batches = 0

    for batch in dataloader:
        if n_batches >= max_batches:
            break

        input_ids = batch["input_ids"].to(device)
        targets = batch["targets"].to(device)

        output = model(input_ids, targets=targets)

        for layer_idx, metrics in enumerate(output["routing_metrics"]):
            routing_entropies[layer_idx] += metrics["routing/entropy"]

        n_batches += 1

    # Average
    routing_entropies /= max(n_batches, 1)

    return {
        "routing_entropies": routing_entropies.numpy(),
        "n_layers": n_layers,
        "n_experts": n_experts,
    }


def plot_routing_entropy(analysis: dict, save_path: str | None = None):
    """Plot routing entropy per layer."""
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = range(analysis["n_layers"])
    entropies = analysis["routing_entropies"]

    ax.bar(layers, entropies, color="steelblue", alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Routing Entropy")
    ax.set_title("Expert Routing Entropy per Layer")
    ax.axhline(
        y=np.log(analysis["n_experts"]),
        color="red",
        linestyle="--",
        label=f"Max entropy (uniform, {analysis['n_experts']} experts)",
    )
    ax.legend()
    ax.set_xticks(layers)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_expert_utilization_heatmap(
    expert_counts: np.ndarray, save_path: str | None = None
):
    """Plot heatmap of expert utilization across layers."""
    # Normalize per layer
    row_sums = expert_counts.sum(axis=1, keepdims=True)
    utilization = expert_counts / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(
        utilization,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=[f"Expert {i}" for i in range(utilization.shape[1])],
        yticklabels=[f"Layer {i}" for i in range(utilization.shape[0])],
        ax=ax,
    )
    ax.set_title("Expert Utilization Heatmap (fraction of tokens per expert)")
    ax.set_xlabel("Expert")
    ax.set_ylabel("Layer")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
