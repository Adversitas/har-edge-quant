import numpy as np
import pytest
import torch

from haredge import int8
from haredge.data import Standardizer, synthetic
from haredge.model import HARNet, TrainConfig, predict_logits, train


@pytest.fixture(scope="module")
def trained():
    tr, va = synthetic(240, seed=1), synthetic(120, seed=2)
    std = Standardizer.fit(tr.x)
    model, _ = train(std(tr.x), tr.y, std(va.x), va.y, TrainConfig(epochs=6, width=8), log=None)
    return model, std(va.x), va.y


def test_quantize_multiplier_roundtrip():
    m = np.array([0.7, 0.01, 1e-4, 0.5])
    m0, shift = int8.quantize_multiplier(m)
    assert np.all(m0 >= 1 << 30) and np.all(m0 < 1 << 31)
    approx = m0 / 2.0 ** (31 + shift)
    np.testing.assert_allclose(approx, m, rtol=1e-8)


def test_quantize_multiplier_rejects_ge_one():
    with pytest.raises(ValueError):
        int8.quantize_multiplier(np.array([1.5]))


def test_requantize_rounds_half_up():
    m0, shift = int8.quantize_multiplier(np.array([0.5]))
    acc = np.array([[3, 4, -3, 101]], dtype=np.int32)
    out = int8.requantize(acc, m0, shift, axis=1)
    np.testing.assert_array_equal(out, [[2, 2, -1, 51]])


def test_fold_matches_torch(trained):
    model, x, _ = trained
    folded = int8.fold_harnet(model)
    ref = predict_logits(model, x)
    np.testing.assert_allclose(int8.float_forward(folded, x), ref, atol=1e-4)


def test_int8_engine_tracks_float(trained):
    model, x, _ = trained
    folded = int8.fold_harnet(model)
    qm = int8.quantize(folded, x[:64])
    fp = int8.float_forward(folded, x).argmax(1)
    q = qm.forward_int(x).argmax(1)
    assert (fp == q).mean() >= 0.9
    sqnr = int8.layer_sqnr(folded, qm, x)
    assert len(sqnr) == 5 and min(sqnr) > 15


def test_int8_engine_is_integer_only(trained):
    model, x, _ = trained
    qm = int8.quantize(int8.fold_harnet(model), x[:32])
    trace = []
    qm.forward_int(x[:4], trace)
    assert all(np.issubdtype(t.dtype, np.integer) for t in trace)
    assert all(q.w_q.dtype == np.int8 for q in qm.layers if q.kind != "gap")
    assert qm.weight_bytes() < sum(p.numel() * 4 for p in model.parameters())


def test_logits_close_to_float(trained):
    model, x, _ = trained
    folded = int8.fold_harnet(model)
    qm = int8.quantize(folded, x[:64])
    fp = int8.float_forward(folded, x)
    q = qm.logits(x)
    rel = np.abs(q - fp).mean() / np.abs(fp).mean()
    assert rel < 0.1
    assert (q.argmax(1) == fp.argmax(1)).mean() >= 0.9


def test_model_shapes():
    m = HARNet(width=8).eval()
    with torch.no_grad():
        out = m(torch.zeros(3, 9, 128))
    assert out.shape == (3, 6)
