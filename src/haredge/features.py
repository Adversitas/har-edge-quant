"""Hand-crafted DSP features: the classical baseline a CNN has to beat.

Per channel (9 raw + 2 magnitudes): time-domain statistics, FFT band energies
on a Hann-windowed spectrum, dominant frequency and spectral entropy. Plus the
mean gravity direction (cheap and very informative for static postures).
"""

from __future__ import annotations

import numpy as np

from .data import SAMPLE_RATE_HZ

BANDS_HZ = [(0.0, 0.5), (0.5, 1.5), (1.5, 3.0), (3.0, 5.0), (5.0, 8.0), (8.0, 12.0), (12.0, 25.0)]


def _with_magnitudes(x: np.ndarray) -> np.ndarray:
    body_mag = np.linalg.norm(x[:, 0:3], axis=1, keepdims=True)
    gyro_mag = np.linalg.norm(x[:, 3:6], axis=1, keepdims=True)
    return np.concatenate([x, body_mag, gyro_mag], axis=1)


def spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-sided power spectrum of each channel, Hann window. Returns (freqs, power)."""
    n = x.shape[-1]
    win = np.hanning(n).astype(np.float32)
    xc = x - x.mean(axis=-1, keepdims=True)
    power = np.abs(np.fft.rfft(xc * win, axis=-1)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / SAMPLE_RATE_HZ)
    return freqs, power


def extract(x: np.ndarray) -> np.ndarray:
    """(n, 9, T) raw windows -> (n, d) feature matrix."""
    xm = _with_magnitudes(x)
    feats = [
        xm.mean(-1),
        xm.std(-1),
        xm.min(-1),
        xm.max(-1),
        np.percentile(xm, 25, axis=-1),
        np.percentile(xm, 75, axis=-1),
        np.abs(np.diff(xm, axis=-1)).mean(-1),  # mean absolute jerk
    ]
    freqs, power = spectrum(xm)
    total = power.sum(-1) + 1e-9
    for lo, hi in BANDS_HZ:
        band = (freqs >= lo) & (freqs < hi)
        feats.append(np.log1p(power[..., band].sum(-1)))
    feats.append(freqs[np.argmax(power[..., 1:], axis=-1) + 1])
    p = power / total[..., None]
    feats.append(-(p * np.log(p + 1e-12)).sum(-1))
    grav = x[:, 6:9].mean(-1)
    grav = grav / (np.linalg.norm(grav, axis=1, keepdims=True) + 1e-9)
    return np.concatenate([np.concatenate(feats, axis=1), grav], axis=1).astype(np.float32)
