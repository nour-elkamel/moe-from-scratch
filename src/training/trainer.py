"""Unified trainer for both dense and MoE transformers."""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from .losses import compute_lm_loss
from .lr_schedule import get_lr


class Trainer:
    """Trainer that handles both dense and MoE models.

    The key difference: MoE models return auxiliary loss (load balancing)
    that must be added to the LM loss during training.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader | None = None,
        lr: float = 3e-4,
        min_lr: float = 3e-5,
        warmup_steps: int = 100,
        max_steps: int = 5000,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        eval_every: int = 250,
        log_every: int = 10,
        save_every: int = 1000,
        save_path: str = "outputs/model",
        use_wandb: bool = True,
        is_moe: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.lr = lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.eval_every = eval_every
        self.log_every = log_every
        self.save_every = save_every
        self.save_path = save_path
        self.use_wandb = use_wandb
        self.is_moe = is_moe

        # Separate weight-decay groups
        decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.95),
        )

    def train(self):
        self.model.train()
        step = 0
        total_loss = 0.0
        total_lm_loss = 0.0
        total_aux_loss = 0.0

        while step < self.max_steps:
            for batch in self.train_dataloader:
                if step >= self.max_steps:
                    break

                # Update learning rate
                lr = get_lr(step, self.lr, self.min_lr, self.warmup_steps, self.max_steps)
                for pg in self.optimizer.param_groups:
                    pg["lr"] = lr

                input_ids = batch["input_ids"].to(next(self.model.parameters()).device)
                targets = batch["targets"].to(next(self.model.parameters()).device)

                output = self.model(input_ids, targets=targets)
                loss = output["loss"]

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                total_lm_loss += output.get("lm_loss", output["loss"]).item()
                if self.is_moe:
                    total_aux_loss += output.get("aux_loss", 0)
                    if isinstance(total_aux_loss, torch.Tensor):
                        total_aux_loss = total_aux_loss.item()

                step += 1

                if step % self.log_every == 0:
                    avg_loss = total_loss / self.log_every
                    avg_lm = total_lm_loss / self.log_every
                    avg_aux = total_aux_loss / self.log_every

                    metrics = {
                        "train/loss": avg_loss,
                        "train/lm_loss": avg_lm,
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "train/step": step,
                    }

                    if self.is_moe:
                        metrics["train/aux_loss"] = avg_aux
                        # Aggregate routing metrics across layers
                        if "routing_metrics" in output:
                            layer_metrics = output["routing_metrics"]
                            avg_entropy = sum(m["routing/entropy"] for m in layer_metrics) / len(layer_metrics)
                            avg_util_std = sum(m["routing/expert_utilization_std"] for m in layer_metrics) / len(layer_metrics)
                            metrics["train/avg_routing_entropy"] = avg_entropy
                            metrics["train/avg_expert_util_std"] = avg_util_std

                    if self.use_wandb:
                        wandb.log(metrics, step=step)
                    ppl = torch.exp(torch.tensor(avg_lm)).item()
                    print(f"Step {step}/{self.max_steps} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {lr:.2e}")

                    total_loss = 0.0
                    total_lm_loss = 0.0
                    total_aux_loss = 0.0

                if step % self.eval_every == 0 and self.eval_dataloader:
                    self._evaluate(step)

                if step % self.save_every == 0:
                    self._save(step)

        self._save(step)

    @torch.no_grad()
    def _evaluate(self, step: int):
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(next(self.model.parameters()).device)
            targets = batch["targets"].to(next(self.model.parameters()).device)

            output = self.model(input_ids, targets=targets)
            lm_loss = output.get("lm_loss", output["loss"])
            total_loss += lm_loss.item()
            n += 1

        avg_loss = total_loss / max(n, 1)
        ppl = torch.exp(torch.tensor(avg_loss)).item()

        metrics = {"eval/loss": avg_loss, "eval/perplexity": ppl}
        if self.use_wandb:
            wandb.log(metrics, step=step)
        print(f"  Eval | Loss: {avg_loss:.4f} | PPL: {ppl:.2f}")

        self.model.train()

    def _save(self, step: int):
        path = f"{self.save_path}/step_{step}"
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        print(f"Saved checkpoint to {path}")
