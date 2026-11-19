"""Integer-only inference engine for HARNet, in plain NumPy.

This is what the network looks like once it leaves PyTorch for a DSP/NPU with
no floating point unit in the inner loop:

* BatchNorm folded into the preceding convolution;
* weights: int8, symmetric, one scale per output channel;
* activations: uint8 after ReLU (zero point 0), int8 for the signed input;
* biases: int32 at scale ``s_in * s_w``;
* accumulation in int32, requantisation by a fixed-point multiplier
  ``M = M0 * 2**-(31 + shift)`` (``M0`` an int32), round-half-up, saturate.

The last layer's int32 accumulators are scaled by ``s_in * s_w[class]`` to give
logits. Because weight scales are per output channel, that one multiply per
class is needed before the argmax (on a target it is 6 multiplies per window).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------- folding


@dataclass
class FloatLayer:
    kind: str  # "conv" | "fc"
    w: np.ndarray  # conv: (O, C, K); fc: (O, C)
    b: np.ndarray  # (O,)
    pad: int = 0
    relu: bool = True
    pool: int = 1  # max-pool factor applied after the activation


def fold_conv_bn(
    w: np.ndarray, gamma, beta, mean, var, eps: float
) -> tuple[np.ndarray, np.ndarray]:
    s = gamma / np.sqrt(var + eps)
    return (w * s[:, None, None]).astype(np.float32), (beta - mean * s).astype(np.float32)


def fold_harnet(model) -> list[FloatLayer]:
    """Extract BN-folded float layers from a trained (eval-mode) HARNet."""
    layers = []
    specs = [("conv1", "bn1", 2), ("conv2", "bn2", 2), ("conv3", "bn3", 1)]
    for conv_name, bn_name, pool in specs:
        conv, bn = getattr(model, conv_name), getattr(model, bn_name)
        w, b = fold_conv_bn(
            conv.weight.detach().numpy(),
            bn.weight.detach().numpy(),
            bn.bias.detach().numpy(),
            bn.running_mean.numpy(),
            bn.running_var.numpy(),
            bn.eps,
        )
        layers.append(FloatLayer("conv", w, b, pad=conv.padding[0], relu=True, pool=pool))
    layers.append(FloatLayer("gap", np.zeros(0), np.zeros(0), relu=False))
    fc = model.fc
    layers.append(
        FloatLayer("fc", fc.weight.detach().numpy(), fc.bias.detach().numpy(), relu=False)
    )
    return layers


# ----------------------------------------------------------------------- float reference


def _im2col(x: np.ndarray, k: int, pad: int) -> np.ndarray:
    """(n, C, T) -> (n, C, K, T) sliding windows with zero padding ('same' conv)."""
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    t = x.shape[-1]
    return np.stack([xp[..., i : i + t] for i in range(k)], axis=2)


def _maxpool(x: np.ndarray, p: int) -> np.ndarray:
    if p == 1:
        return x
    n, c, t = x.shape
    return x[..., : t - t % p].reshape(n, c, t // p, p).max(-1)


def float_forward(layers: list[FloatLayer], x: np.ndarray, trace: list | None = None) -> np.ndarray:
    """Float reference of the folded network (should match PyTorch to ~1e-5)."""
    h = x.astype(np.float32)
    for layer in layers:
        if layer.kind == "conv":
            cols = _im2col(h, layer.w.shape[2], layer.pad)
            h = np.einsum("nckt,ock->not", cols, layer.w, optimize=True) + layer.b[None, :, None]
            if layer.relu:
                h = np.maximum(h, 0)
            h = _maxpool(h, layer.pool)
        elif layer.kind == "gap":
            h = h.mean(-1)
        else:
            h = h @ layer.w.T + layer.b
        if trace is not None:
            trace.append(h)
    return h


# --------------------------------------------------------------------- quantisation


def quantize_multiplier(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real multiplier(s) in (0, 1) -> (M0 int32 in [2^30, 2^31), right shift >= 0)."""
    m = np.asarray(m, dtype=np.float64)
    mant, exp = np.frexp(m)  # m = mant * 2**exp, mant in [0.5, 1)
    m0 = np.round(mant * (1 << 31)).astype(np.int64)
    fix = m0 == (1 << 31)
    m0[fix] //= 2
    exp = exp + fix
    shift = -exp
    if np.any(shift < 0):
        raise ValueError("requantisation multiplier >= 1 is not supported")
    return m0, shift.astype(np.int64)


def requantize(acc: np.ndarray, m0: np.ndarray, shift: np.ndarray, axis: int = 1) -> np.ndarray:
    """int32 accumulators * M0 >> (31 + shift), rounded half up, in int64 arithmetic."""
    shape = [1] * acc.ndim
    shape[axis] = -1
    m0 = np.broadcast_to(m0, acc.shape[axis : axis + 1]).reshape(shape)
    total_shift = 31 + np.broadcast_to(shift, acc.shape[axis : axis + 1]).reshape(shape)
    prod = acc.astype(np.int64) * m0
    return (prod + (np.int64(1) << (total_shift - 1))) >> total_shift


@dataclass
class QLayer:
    kind: str
    w_q: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int8))
    b_q: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int32))
    w_scale: np.ndarray = field(default_factory=lambda: np.zeros(0))
    m0: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    shift: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    in_scale: float = 1.0
    out_scale: float = 1.0
    pad: int = 0
    pool: int = 1
    relu: bool = True


@dataclass
class Int8Model:
    in_scale: float
    layers: list[QLayer]

    def weight_bytes(self) -> int:
        return int(sum(q.w_q.nbytes + q.b_q.nbytes for q in self.layers))

    def quantize_input(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.round(x / self.in_scale), -127, 127).astype(np.int8)

    def forward_int(self, x: np.ndarray, trace: list | None = None) -> np.ndarray:
        """Integer-only path. Returns int32 accumulators of the last layer."""
        h = self.quantize_input(x).astype(np.int32)
        for q in self.layers:
            if q.kind == "conv":
                cols = _im2col(h, q.w_q.shape[2], q.pad)
                acc = np.einsum("nckt,ock->not", cols, q.w_q.astype(np.int32), optimize=True)
                acc = acc + q.b_q[None, :, None]
                out = requantize(acc, q.m0, q.shift, axis=1)
                lo = 0 if q.relu else -127
                h = _maxpool(np.clip(out, lo, 255).astype(np.int32), q.pool)
            elif q.kind == "gap":
                acc = h.sum(-1)  # int32 sum, the 1/T is folded into the multiplier
                h = np.clip(requantize(acc, q.m0, q.shift, axis=1), 0, 255).astype(np.int32)
            else:
                h = h @ q.w_q.astype(np.int32).T + q.b_q[None, :]
            if trace is not None:
                trace.append(h)
        return h

    def logits(self, x: np.ndarray) -> np.ndarray:
        acc = self.forward_int(x)
        last = self.layers[-1]
        return acc.astype(np.float64) * (last.in_scale * last.w_scale)[None, :]


def _calib_max(a: np.ndarray, percentile: float) -> float:
    v = np.abs(a).ravel()
    return float(np.percentile(v, percentile)) if percentile < 100 else float(v.max())


def quantize(layers: list[FloatLayer], x_calib: np.ndarray, percentile: float = 99.99) -> Int8Model:
    """Post-training static quantisation calibrated on ``x_calib`` (float, standardised)."""
    trace: list[np.ndarray] = []
    float_forward(layers, x_calib, trace)
    in_scale = _calib_max(x_calib, percentile) / 127.0
    s_in = in_scale
    qlayers = []
    for layer, act in zip(layers, trace, strict=True):
        if layer.kind == "gap":
            t_len = trace[len(qlayers) - 1].shape[-1]
            s_out = _calib_max(act, percentile) / 255.0
            m0, shift = quantize_multiplier(np.array([s_in / (t_len * s_out)]))
            qlayers.append(QLayer("gap", m0=m0, shift=shift, in_scale=s_in, out_scale=s_out))
            s_in = s_out
            continue
        w = layer.w
        w_max = np.abs(w.reshape(w.shape[0], -1)).max(1)
        w_scale = np.maximum(w_max, 1e-12) / 127.0
        shape = (-1,) + (1,) * (w.ndim - 1)
        w_q = np.clip(np.round(w / w_scale.reshape(shape)), -127, 127).astype(np.int8)
        b_q = np.round(layer.b / (s_in * w_scale)).astype(np.int32)
        if layer.kind == "fc":
            qlayers.append(QLayer("fc", w_q, b_q, w_scale, in_scale=s_in, relu=False))
            continue
        s_out = _calib_max(act, percentile) / 255.0
        m0, shift = quantize_multiplier(s_in * w_scale / s_out)
        qlayers.append(
            QLayer(
                "conv", w_q, b_q, w_scale, m0, shift, s_in, s_out, layer.pad, layer.pool, layer.relu
            )
        )
        s_in = s_out
    return Int8Model(in_scale=in_scale, layers=qlayers)


def layer_sqnr(layers: list[FloatLayer], qmodel: Int8Model, x: np.ndarray) -> list[float]:
    """Signal-to-quantisation-noise ratio (dB) of each layer's output vs the float path."""
    ft: list[np.ndarray] = []
    it: list[np.ndarray] = []
    float_forward(layers, x, ft)
    qmodel.forward_int(x, it)
    out = []
    for f, i, q in zip(ft, it, qmodel.layers, strict=True):
        scale = q.in_scale * q.w_scale[None, :] if q.kind == "fc" else q.out_scale
        deq = i.astype(np.float64) * scale
        noise = np.mean((f - deq) ** 2)
        out.append(float(10 * np.log10(np.mean(f**2) / max(noise, 1e-20))))
    return out
