# -*- coding: utf-8 -*-
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import roc_auc_score
except Exception:
    roc_auc_score = None


class DeepFM(nn.Module):
    def __init__(
        self,
        feature_sizes: List[int],
        embedding_size: int = 8,
        hidden_dims: List[int] = None,
        dropout: List[float] = None,
        use_cuda: bool = False,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        if dropout is None:
            dropout = [0.2, 0.2]

        self.field_size = len(feature_sizes)
        self.emb_size = embedding_size
        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

        self.emb_1st = nn.ModuleList([nn.Embedding(s, 1) for s in feature_sizes])
        self.emb_2nd = nn.ModuleList([nn.Embedding(s, embedding_size) for s in feature_sizes])

        in_dim = self.field_size * embedding_size
        layers = []
        for i, dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout[min(i, len(dropout) - 1)]))
            in_dim = dim
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(in_dim, 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def _get_embeddings(self, xi: torch.Tensor, xv: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        weighted_embs = []
        for i in range(self.field_size):
            weighted_embs.append(self.emb_2nd[i](xi[:, i]) * xv[:, i].unsqueeze(-1))
        return torch.stack(weighted_embs, dim=1), weighted_embs

    def get_representation(self, xi: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
        xi = xi.long()
        xv = xv.float()
        emb_stack, weighted_embs = self._get_embeddings(xi, xv)
        fm_sum = emb_stack.sum(dim=1)
        deep_in = torch.cat(weighted_embs, dim=1)
        return torch.cat([fm_sum, deep_in], dim=1)

    def forward(self, xi: torch.Tensor, xv: torch.Tensor) -> torch.Tensor:
        xi = xi.long()
        xv = xv.float()

        first = [self.emb_1st[i](xi[:, i]) * xv[:, i].unsqueeze(-1) for i in range(self.field_size)]
        first = torch.cat(first, dim=1)

        emb_stack, weighted_embs = self._get_embeddings(xi, xv)
        sum_e = emb_stack.sum(dim=1)
        second = 0.5 * (sum_e.pow(2) - emb_stack.pow(2).sum(dim=1))

        deep_in = torch.cat(weighted_embs, dim=1)
        d = self.deep_out(self.deep(deep_in))

        out = first.sum(1, keepdim=True) + second.sum(1, keepdim=True) + d + self.bias
        return out.squeeze(1)

    def _compute_auc(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        if roc_auc_score is None:
            return float("nan")
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))

    def evaluate(self, loader) -> Dict[str, float]:
        self.eval()
        criterion = F.binary_cross_entropy_with_logits
        losses, y_true, y_prob = [], [], []

        with torch.no_grad():
            for xi, xv, y in loader:
                xi = xi.to(self.device, non_blocking=True)
                xv = xv.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).float()

                logits = self(xi, xv)
                loss = criterion(logits, y)
                prob = torch.sigmoid(logits)

                losses.append(float(loss.item()))
                y_true.append(y.cpu().numpy())
                y_prob.append(prob.cpu().numpy())

        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        y_pred = (y_prob >= 0.5).astype(np.float32)
        acc = float((y_pred == y_true).mean())
        auc = self._compute_auc(y_true, y_prob)
        return {
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "acc": acc,
            "auc": auc,
        }

    def fit(
        self,
        loader_train,
        loader_val,
        optimizer,
        epochs: int = 30,
        scheduler=None,
        model_path: str = "best_deepfm.pt",
        early_stop_patience: int = 8,
    ) -> List[Dict[str, float]]:
        self.to(self.device)
        criterion = F.binary_cross_entropy_with_logits

        best_auc = -1.0
        best_epoch = -1
        no_improve = 0
        history = []

        for ep in range(epochs):
            t0 = time.time()
            self.train()
            train_losses = []

            for xi, xv, y in loader_train:
                xi = xi.to(self.device, non_blocking=True)
                xv = xv.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).float()

                logits = self(xi, xv)
                loss = criterion(logits, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item()))

            if scheduler is not None:
                scheduler.step()

            metrics = self.evaluate(loader_val)
            epoch_time = time.time() - t0
            item = {
                "epoch": ep + 1,
                "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
                "val_loss": metrics["loss"],
                "val_acc": metrics["acc"],
                "val_auc": metrics["auc"],
                "epoch_time_sec": epoch_time,
            }
            history.append(item)

            print(
                f"Epoch {ep+1:02d}/{epochs} | "
                f"time={epoch_time:.3f}s | "
                f"train_loss={item['train_loss']:.4f} | "
                f"val_loss={item['val_loss']:.4f} | "
                f"val_acc={item['val_acc']:.4f} | "
                f"val_auc={item['val_auc']:.6f}"
            )

            current_auc = item["val_auc"]
            if not np.isnan(current_auc) and current_auc > best_auc:
                best_auc = current_auc
                best_epoch = ep + 1
                no_improve = 0
                torch.save(self.state_dict(), model_path)
            else:
                no_improve += 1

            if no_improve >= early_stop_patience:
                print(f"Early stop at epoch {ep+1}, best epoch={best_epoch}, best auc={best_auc:.6f}")
                break

        if best_epoch > 0:
            self.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded best model from epoch {best_epoch}, best auc={best_auc:.6f}")
        return history

    def predict_proba(self, loader) -> np.ndarray:
        self.eval()
        probs = []
        with torch.no_grad():
            for batch in loader:
                if len(batch) == 3:
                    xi, xv, _ = batch
                else:
                    xi, xv = batch
                xi = xi.to(self.device, non_blocking=True)
                xv = xv.to(self.device, non_blocking=True)
                p = torch.sigmoid(self(xi, xv))
                probs.append(p.cpu().numpy())
        if not probs:
            return np.array([], dtype=np.float32)
        return np.concatenate(probs)
