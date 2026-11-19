"""UCI HAR loader: raw inertial windows (9 channels x 128 samples at 50 Hz).

Dataset: Anguita et al., "A Public Domain Dataset for Human Activity Recognition
Using Smartphones", ESANN 2013. 30 subjects, 6 activities, subject-disjoint
train (21 subjects) / test (9 subjects) split provided by the authors.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)

CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]
ACTIVITIES = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
SAMPLE_RATE_HZ = 50
WINDOW = 128


@dataclass
class Split:
    x: np.ndarray  # (n, 9, 128) float32
    y: np.ndarray  # (n,) int64, 0..5
    subject: np.ndarray  # (n,) int64


def _read_matrix(raw: bytes) -> np.ndarray:
    return np.loadtxt(io.BytesIO(raw), dtype=np.float32)


def _parse_split(zf: zipfile.ZipFile, root: str, split: str) -> Split:
    chans = []
    for ch in CHANNELS:
        name = f"{root}/{split}/Inertial Signals/{ch}_{split}.txt"
        chans.append(_read_matrix(zf.read(name)))
    x = np.stack(chans, axis=1)
    y = _read_matrix(zf.read(f"{root}/{split}/y_{split}.txt")).astype(np.int64) - 1
    s = _read_matrix(zf.read(f"{root}/{split}/subject_{split}.txt")).astype(np.int64)
    return Split(x=x, y=y, subject=s)


def download(cache_dir: Path) -> Path:
    """Download the archive once and convert it to a compact .npz cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / "uci_har.npz"
    if out.exists():
        return out
    with urllib.request.urlopen(URL, timeout=120) as resp:
        outer = zipfile.ZipFile(io.BytesIO(resp.read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("UCI HAR Dataset.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    train = _parse_split(inner, "UCI HAR Dataset", "train")
    test = _parse_split(inner, "UCI HAR Dataset", "test")
    np.savez_compressed(
        out,
        x_train=train.x,
        y_train=train.y,
        s_train=train.subject,
        x_test=test.x,
        y_test=test.y,
        s_test=test.subject,
    )
    return out


def load(cache_dir: Path) -> tuple[Split, Split]:
    d = np.load(download(cache_dir))
    return (
        Split(d["x_train"], d["y_train"], d["s_train"]),
        Split(d["x_test"], d["y_test"], d["s_test"]),
    )


def subject_holdout(split: Split, n_val_subjects: int = 4, seed: int = 0) -> tuple[Split, Split]:
    """Carve a validation set out of train by whole subjects (no window leakage)."""
    rng = np.random.default_rng(seed)
    subjects = np.unique(split.subject)
    val_subj = rng.choice(subjects, size=n_val_subjects, replace=False)
    m = np.isin(split.subject, val_subj)
    return (
        Split(split.x[~m], split.y[~m], split.subject[~m]),
        Split(split.x[m], split.y[m], split.subject[m]),
    )


@dataclass
class Standardizer:
    mean: np.ndarray  # (9, 1)
    std: np.ndarray  # (9, 1)

    @classmethod
    def fit(cls, x: np.ndarray) -> Standardizer:
        mean = x.mean(axis=(0, 2), keepdims=True)[0]
        std = x.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        return cls(mean.astype(np.float32), std.astype(np.float32))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)


def synthetic(n: int = 240, seed: int = 0) -> Split:
    """Small synthetic stand-in with class-dependent frequencies (tests and CI only)."""
    rng = np.random.default_rng(seed)
    t = np.arange(WINDOW) / SAMPLE_RATE_HZ
    y = rng.integers(0, 6, size=n)
    x = np.empty((n, 9, WINDOW), dtype=np.float32)
    for i, c in enumerate(y):
        f = 0.5 + 0.6 * c
        amp = 1.0 if c < 3 else 0.1
        base = amp * np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        grav = np.zeros(9)
        grav[6 + (c % 3)] = 1.0
        x[i] = base[None, :] * rng.uniform(0.5, 1.5, size=(9, 1)) + grav[:, None]
        x[i] += 0.05 * rng.standard_normal((9, WINDOW))
    return Split(x, y.astype(np.int64), np.arange(n) % 10)
