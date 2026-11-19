"""Compression variants built with PyTorch tooling: int8 PTQ and magnitude pruning."""

from __future__ import annotations

import copy
import io
import time

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune

from .model import HARNet


def serialized_bytes(model: nn.Module) -> int:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


def ptq_int8(model: HARNet, x_calib: np.ndarray, backend: str = "x86") -> nn.Module:
    """Eager-mode post-training static quantisation (per-channel int8 weights)."""
    torch.backends.quantized.engine = backend
    m = model.fused()
    m.qconfig = torch.ao.quantization.get_default_qconfig(backend)
    torch.ao.quantization.prepare(m, inplace=True)
    with torch.no_grad():
        for i in range(0, len(x_calib), 256):
            m(torch.from_numpy(x_calib[i : i + 256]))
    torch.ao.quantization.convert(m, inplace=True)
    return m


def magnitude_prune(model: HARNet, amount: float) -> HARNet:
    """Global unstructured L1 pruning of conv/linear weights (made permanent)."""
    m = copy.deepcopy(model)
    params = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv1d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for mod, name in params:
        prune.remove(mod, name)
    return m.eval()


def sparsity(model: nn.Module) -> float:
    ws = [mod.weight for mod in model.modules() if isinstance(mod, (nn.Conv1d, nn.Linear))]
    total = sum(w.numel() for w in ws)
    zeros = sum(int((w == 0).sum()) for w in ws)
    return zeros / total


@torch.no_grad()
def latency_ms(model: nn.Module, x_one: np.ndarray, n: int = 300, warmup: int = 30) -> float:
    """Median single-window CPU latency (batch 1, one thread), in milliseconds."""
    torch.set_num_threads(1)
    t = torch.from_numpy(x_one[None])
    for _ in range(warmup):
        model(t)
    times = []
    for _ in range(n):
        s = time.perf_counter()
        model(t)
        times.append(time.perf_counter() - s)
    return 1000 * float(np.median(times))
