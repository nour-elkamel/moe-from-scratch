"""Analyze and visualize expert routing patterns in a trained MoE model."""

import argparse
import torch
from omegaconf import OmegaConf

from src.models import MoETransformer
from src.data import build_dataloader
from src.evaluation.routing_analysis import analyze_routing, plot_routing_entropy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/moe_top2.yaml")
    parser.add_argument("--output_dir", type=str, default="outputs/analysis")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    model = MoETransformer(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        num_experts=cfg.model.num_experts,
        top_k=cfg.model.top_k,
        d_ff=cfg.model.get("d_ff"),
        max_seq_len=cfg.model.max_seq_len,
        dropout=0.0,  # No dropout during eval
        load_balance_coeff=cfg.model.get("load_balance_coeff", 0.01),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.load_state_dict(torch.load(f"{args.checkpoint}/model.pt", map_location=device))
    model = model.to(device)

    eval_dl = build_dataloader(
        dataset_name=cfg.data.dataset,
        split="validation",
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        max_samples=1000,
    )

    print("Analyzing routing patterns...")
    analysis = analyze_routing(model, eval_dl, device=device)

    import os
    os.makedirs(args.output_dir, exist_ok=True)

    plot_routing_entropy(analysis, save_path=f"{args.output_dir}/routing_entropy.png")
    print(f"Saved routing analysis to {args.output_dir}/")


if __name__ == "__main__":
    main()
