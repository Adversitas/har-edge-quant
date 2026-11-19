"""A small 1D CNN for 9x128 inertial windows, written to be quantization-friendly.

Conv -> BatchNorm -> ReLU blocks (fusable), max pooling, global average pooling,
one linear head. Quant/DeQuant stubs mark the int8 region for eager-mode PTQ.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

N_CLASSES = 6


class HARNet(nn.Module):
    def __init__(self, in_ch: int = 9, width: int = 32, n_classes: int = N_CLASSES):
        super().__init__()
        w1, w2, w3 = width, 2 * width, 2 * width
        self.quant = torch.ao.quantization.QuantStub()
        self.conv1 = nn.Conv1d(in_ch, w1, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(w1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(w1, w2, 5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(w2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(w2, w3, 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(w3)
        self.relu3 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(w3, n_classes)
        self.dequant = torch.ao.quantization.DeQuantStub()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.gap(x).flatten(1)
        return self.dequant(self.fc(x))

    def fused(self) -> HARNet:
        """Eval-mode copy with Conv+BN+ReLU fused (the graph an int8 kernel runs)."""
        m = copy.deepcopy(self).eval()
        torch.ao.quantization.fuse_modules(
            m,
            [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"], ["conv3", "bn3", "relu3"]],
            inplace=True,
        )
        return m


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    seed: int = 0
    width: int = 32


def _augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Light, physically plausible augmentation: amplitude jitter and time shift."""
    scale = 1.0 + 0.1 * torch.randn(x.shape[0], x.shape[1], 1, generator=gen)
    shift = int(torch.randint(-8, 9, (1,), generator=gen))
    return torch.roll(x * scale, shifts=shift, dims=-1)


def train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    cfg: TrainConfig,
    log=print,
) -> tuple[HARNet, list[dict]]:
    """Train with AdamW + cosine schedule; keep the best epoch on validation macro-F1."""
    from sklearn.metrics import f1_score

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = HARNet(in_ch=x_train.shape[1], width=cfg.width)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    xt, yt = torch.from_numpy(x_train), torch.from_numpy(y_train)
    xv = torch.from_numpy(x_val)
    best, best_state, history = -1.0, None, []
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(xt), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            loss = nn.functional.cross_entropy(model(_augment(xt[idx], gen)), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv).argmax(1).numpy()
        f1 = float(f1_score(y_val, pred, average="macro"))
        history.append({"epoch": epoch + 1, "loss": total / len(xt), "val_macro_f1": f1})
        if f1 > best:
            best, best_state = f1, copy.deepcopy(model.state_dict())
        if log and ((epoch + 1) % 5 == 0 or epoch == 0):
            log(f"epoch {epoch + 1:3d}  loss {total / len(xt):.4f}  val macro-F1 {f1:.4f}")
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval(), history


@torch.no_grad()
def predict_logits(model: nn.Module, x: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = [model(torch.from_numpy(x[i : i + batch])).numpy() for i in range(0, len(x), batch)]
    return np.concatenate(out)
