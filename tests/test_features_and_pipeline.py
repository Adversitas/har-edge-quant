import json

import numpy as np

from haredge import compress, features
from haredge.data import Standardizer, subject_holdout, synthetic
from haredge.experiment import run
from haredge.model import HARNet


def test_spectrum_finds_dominant_frequency():
    t = np.arange(128) / 50
    x = np.sin(2 * np.pi * 5.0 * t)[None, None, :].astype(np.float32)
    freqs, power = features.spectrum(x)
    assert abs(freqs[power[0, 0].argmax()] - 5.0) < 0.4


def test_feature_matrix_shape_and_finite():
    s = synthetic(20)
    f = features.extract(s.x)
    assert f.shape[0] == 20 and np.isfinite(f).all()


def test_subject_holdout_is_disjoint():
    s = synthetic(100)
    tr, va = subject_holdout(s, n_val_subjects=3)
    assert not set(tr.subject) & set(va.subject)
    assert len(tr.y) + len(va.y) == 100


def test_standardizer_zero_mean():
    s = synthetic(50)
    z = Standardizer.fit(s.x)(s.x)
    assert np.allclose(z.mean(axis=(0, 2)), 0, atol=1e-4)


def test_pruning_reaches_target_sparsity():
    m = compress.magnitude_prune(HARNet(width=8), 0.75)
    assert abs(compress.sparsity(m) - 0.75) < 0.01


def test_end_to_end_synthetic(tmp_path):
    res = run(
        tmp_path, tmp_path / "rep", seeds=[0], epochs=2, use_synthetic=True, log=lambda *_: None
    )
    assert (tmp_path / "rep" / "results.md").exists()
    saved = json.loads((tmp_path / "rep" / "results.json").read_text())
    assert "fp32" in saved["summary"] and "int8_engine_p99.99" in saved["summary"]
    assert (
        res["summary"]["int8_engine_p99.99"]["bytes"]["mean"]
        < res["summary"]["fp32"]["bytes"]["mean"]
    )
