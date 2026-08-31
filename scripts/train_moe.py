"""Train MoE transformer."""

import argparse
import torch
import wandb
from omegaconf import OmegaConf

from src.models import MoETransformer
from src.data import build_dataloader
from src.training import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/moe_top2.yaml")
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
        capacity_factor=cfg.model.get("capacity_factor", 1.25),
        max_seq_len=cfg.model.max_seq_len,
        dropout=cfg.model.dropout,
        load_balance_coeff=cfg.model.get("load_balance_coeff", 0.01),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    params = model.count_parameters()
    print(f"MoE model: {params['total_million']:.1f}M total, {params['active_million']:.1f}M active per token")

    wandb.init(
        project="moe-from-scratch",
        name=f"moe-{cfg.model.num_experts}E-top{cfg.model.top_k}-{params['total_million']:.0f}M",
        config=OmegaConf.to_container(cfg),
    )

    train_dl = build_dataloader(
        dataset_name=cfg.data.dataset,
        split="train",
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        max_samples=cfg.data.get("max_samples"),
    )
    eval_dl = build_dataloader(
        dataset_name=cfg.data.dataset,
        split="validation",
        seq_len=cfg.model.max_seq_len,
        batch_size=cfg.training.batch_size,
        max_samples=cfg.data.get("max_eval_samples", 1000),
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        eval_dataloader=eval_dl,
        lr=cfg.training.lr,
        min_lr=cfg.training.get("min_lr", cfg.training.lr / 10),
        warmup_steps=cfg.training.warmup_steps,
        max_steps=cfg.training.max_steps,
        eval_every=cfg.training.eval_every,
        save_path=cfg.training.save_path,
        is_moe=True,
    )

    trainer.train()
    wandb.finish()


if __name__ == "__main__":
    main()
