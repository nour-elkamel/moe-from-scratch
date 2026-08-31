# Mixture-of-Experts from Scratch

A small Mixture-of-Experts (MoE) transformer built entirely from scratch in PyTorch. Implements expert routing, load balancing losses, and capacity factors — then trains on language modeling to compare against a dense baseline of equivalent compute.

## Why Build MoE from Scratch?

MoE is in every frontier model — Mixtral, GPT-4, Gemini, DeepSeek-V3 — but almost nobody has built one from first principles. Understanding expert routing, load balancing, and the interaction between sparsity and training dynamics is essential for anyone working on next-generation architectures.

## What's Implemented

| Component | Details |
|-----------|---------|
| **Dense Transformer** | Standard GPT-2 style decoder-only transformer (baseline) |
| **Top-k Router** | Softmax gating with top-1 and top-2 expert selection |
| **Expert FFN** | Per-expert feed-forward networks replacing the dense FFN |
| **Load Balancing Loss** | Auxiliary loss to prevent expert collapse (Switch Transformer style) |
| **Capacity Factor** | Controls max tokens per expert per batch |
| **Z-Loss** | Router logit regularization (from ST-MoE) |
| **Expert Parallelism** | Optional multi-GPU expert sharding |

## Project Structure

```
moe-from-scratch/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py         # Standard dense transformer (baseline)
│   │   ├── moe_transformer.py     # MoE transformer
│   │   ├── attention.py           # Multi-head attention
│   │   ├── ffn.py                 # Dense FFN
│   │   ├── moe_layer.py           # MoE layer: router + experts
│   │   ├── router.py              # Top-k gating / routing logic
│   │   └── experts.py             # Expert FFN collection
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py             # Training loop with MoE-specific logging
│   │   ├── losses.py              # LM loss + load balancing + z-loss
│   │   └── lr_schedule.py         # Warmup + cosine decay
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py             # Language modeling dataset (TinyStories / OpenWebText)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── perplexity.py          # Perplexity evaluation
│   │   └── routing_analysis.py    # Expert utilization & routing visualizations
│   └── utils/
│       ├── __init__.py
│       ├── config.py              # Model & training configs
│       └── logging.py             # W&B integration
├── configs/
│   ├── dense_baseline.yaml        # Dense transformer config
│   ├── moe_top1.yaml              # MoE with top-1 routing
│   ├── moe_top2.yaml              # MoE with top-2 routing
│   └── moe_ablations.yaml         # Ablation sweep config
├── scripts/
│   ├── train_dense.py             # Train dense baseline
│   ├── train_moe.py               # Train MoE model
│   ├── evaluate.py                # Run perplexity eval
│   ├── analyze_routing.py         # Visualize expert routing patterns
│   └── run_ablations.sh           # Full ablation sweep
├── notebooks/
│   ├── 01_moe_architecture.ipynb  # Architecture walkthrough + diagrams
│   ├── 02_routing_analysis.ipynb  # Expert utilization visualizations
│   └── 03_dense_vs_moe.ipynb      # Comparison & ablation results
├── outputs/                        # Checkpoints, logs (gitignored)
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/nour-elkamel/moe-from-scratch.git
cd moe-from-scratch
pip install -e .
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1
- Single GPU >= 16GB (A100/4090). Multi-GPU optional for expert parallelism experiments.

## Quick Start

### 1. Train dense baseline

```bash
python scripts/train_dense.py --config configs/dense_baseline.yaml
```

### 2. Train MoE model

```bash
python scripts/train_moe.py --config configs/moe_top2.yaml
```

### 3. Analyze expert routing

```bash
python scripts/analyze_routing.py --checkpoint outputs/moe_top2/final
```

## Model Configurations

Controlled comparison: **same total parameters, same compute budget (FLOPs)**.

| Config | Params (total) | Params (active) | Experts | Top-k | Layers |
|--------|---------------|-----------------|---------|-------|--------|
| Dense baseline | 85M | 85M | 1 | — | 12 |
| MoE-8E Top-1 | 250M | ~85M | 8 | 1 | 12 |
| MoE-8E Top-2 | 250M | ~120M | 8 | 2 | 12 |
| MoE-16E Top-2 | 420M | ~120M | 16 | 2 | 12 |

The MoE models have more total parameters but similar **active parameters** per forward pass — this is the core MoE trade-off.

## Dataset

[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) for fast iteration, [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) for real-world evaluation. Custom tokenizer or GPT-2 tokenizer.

## The Router: How It Works

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, num_experts, top_k, capacity_factor=1.25):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.top_k = top_k
        self.capacity_factor = capacity_factor

    def forward(self, x):
        # x: (batch * seq_len, d_model)
        logits = self.gate(x)                          # (tokens, num_experts)
        scores = F.softmax(logits, dim=-1)             # routing probabilities
        top_k_scores, top_k_indices = scores.topk(self.top_k, dim=-1)

        # Capacity: max tokens each expert can process
        capacity = int(self.capacity_factor * x.size(0) / self.num_experts * self.top_k)

        # Compute load balancing loss
        load_balance_loss = self._load_balance_loss(scores, top_k_indices)

        return top_k_scores, top_k_indices, load_balance_loss
```

## Load Balancing Loss

Expert collapse (all tokens routed to 1-2 experts) is the main failure mode. We implement:

```python
def load_balance_loss(router_probs, expert_indices, num_experts):
    """
    Switch Transformer load balancing loss.
    Encourages uniform expert utilization.
    """
    # Fraction of tokens routed to each expert
    tokens_per_expert = torch.zeros(num_experts, device=router_probs.device)
    tokens_per_expert.scatter_add_(0, expert_indices.view(-1),
                                    torch.ones_like(expert_indices.view(-1), dtype=torch.float))
    f = tokens_per_expert / expert_indices.numel()

    # Mean routing probability for each expert
    p = router_probs.mean(dim=0)

    # Loss: dot product encourages f and p to be uniform
    return num_experts * (f * p).sum()
```

## Ablation Studies

| Variable | Values | Purpose |
|----------|--------|---------|
| **Num experts** | 4, 8, 16, 32 | Sparsity vs capacity trade-off |
| **Top-k** | 1, 2, 4 | Active compute per token |
| **Capacity factor** | 1.0, 1.25, 1.5, 2.0 | Token dropping vs memory |
| **Load balance coeff** | 0.001, 0.01, 0.1 | Routing uniformity pressure |
| **Z-loss coeff** | 0, 0.001, 0.01 | Router logit regularization |
| **Expert init** | Same, different seeds | Does initialization affect specialization? |

## Analysis & Visualizations

Beyond perplexity, we produce:

- **Expert utilization heatmap** — tokens per expert across layers and training steps
- **Routing entropy** — per-layer routing diversity (high = uniform, low = collapsed)
- **Expert specialization** — what patterns does each expert learn? (via input token analysis)
- **Token dropping rate** — how many tokens exceed capacity and get dropped?
- **Training dynamics** — loss curves, load balance loss, gradient norms

## Expected Results

| Model | Perplexity | Active FLOPs | Total Params | Expert Entropy |
|-------|-----------|--------------|-------------|----------------|
| Dense (85M) | — | baseline | 85M | N/A |
| MoE-8E Top-1 | — | ~1x | 250M | — |
| MoE-8E Top-2 | — | ~1.4x | 250M | — |
| MoE-16E Top-2 | — | ~1.4x | 420M | — |

Key hypothesis: MoE-8E Top-2 should match or beat the dense baseline in perplexity while using similar active compute — demonstrating the fundamental MoE benefit.

## References

- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity" (2022)
- Lepikhin et al., "GShard: Scaling Giant Models with Conditional Computation" (2021)
- Zoph et al., "ST-MoE: Designing Stable and Transferable Sparse Expert Models" (2022)
- Jiang et al., "Mixtral of Experts" (2024)
- Dai et al., "DeepSeekMoE: Towards Ultimate Expert Specialization" (2024)

## License

MIT
