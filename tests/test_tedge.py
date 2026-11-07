import numpy as np
import pytest
import torch

from tedge import data, features, model


@pytest.fixture(scope="module")
def ds():
    return data.generate(n_devices=10, windows_per_device=40, seed=3)


def test_shapes_and_labels(ds):
    assert ds["x"].shape == (400, 6, data.N)
    assert ds["x"].dtype == np.float32
    assert set(np.unique(ds["y"])) <= set(range(len(data.CLASSES)))


def test_gravity_is_about_one_g_whatever_the_mounting(ds):
    g = np.linalg.norm(ds["x"][:, :3].mean(axis=2), axis=1)
    assert np.all(np.abs(g - data.G) < 1.0)


def test_labels_follow_thresholds(ds):
    for kind, cls in (("braking", 1), ("acceleration", 2), ("cornering", 3), ("bump", 4)):
        m = ds["kind"] == kind
        harsh = ds["peak"][m] >= data.THRESHOLDS[kind]
        assert np.all((ds["y"][m] == cls) == harsh)
    assert np.all(ds["y"][ds["kind"] == "cruise"] == 0)


def test_device_split_has_no_shared_boxes(ds):
    tr, te = data.split_by_device(ds)
    assert not set(tr["device"]) & set(te["device"])
    assert len(tr["y"]) + len(te["y"]) == len(ds["y"])


def test_features_do_not_depend_on_mounting(ds):
    x = ds["x"][:16]
    rot = model.random_rotation(np.random.default_rng(0), len(x))
    f1, _ = features.features(x)
    f2, _ = features.features(model.rotate_batch(x, rot))
    np.testing.assert_allclose(f1, f2, rtol=1e-3, atol=1e-3)


def test_random_rotations_are_proper():
    r = model.random_rotation(np.random.default_rng(1), 50)
    np.testing.assert_allclose(r @ r.transpose(0, 2, 1), np.broadcast_to(np.eye(3), r.shape), atol=1e-9)
    np.testing.assert_allclose(np.linalg.det(r), 1.0, atol=1e-9)


def test_torch_invariant_matches_numpy(ds):
    x = ds["x"][:8]
    c = features.decompose(x)
    t = model.to_invariant(x)
    for i, k in enumerate(["vert", "h_mag", "yaw", "vib"]):
        np.testing.assert_allclose(t[:, i], c[k], atol=1e-4)


def test_invariant_model_ignores_mounting(ds):
    torch.manual_seed(0)
    m = model.EdgeModel(model.IMUNet(in_channels=4), torch.ones(4), invariant=True).eval()
    x = ds["x"][:8]
    xr = model.rotate_batch(x, model.random_rotation(np.random.default_rng(2), len(x)))
    np.testing.assert_allclose(model.predict_logits(m, x), model.predict_logits(m, xr), atol=1e-3)


def test_rule_baseline_never_says_acceleration(ds):
    assert 2 not in features.rule_baseline(ds["x"])


def test_training_beats_majority_and_onnx_matches(ds, tmp_path):
    pytest.importorskip("onnxruntime")
    from tedge import edge

    tr, te = data.split_by_device(ds)
    m = model.train(tr["x"], tr["y"], epochs=3, log=lambda *_: None)
    p = model.predict(m, te["x"])
    majority = np.bincount(te["y"]).max() / len(te["y"])
    assert (p == te["y"]).mean() > majority
    path = edge.export_onnx(m, tmp_path / "m.onnx")
    np.testing.assert_allclose(edge.onnx_predict(path, te["x"]), model.predict_logits(m, te["x"]), atol=1e-3)
    q = edge.quantize_int8(path, tmp_path / "m8.onnx", tr["x"][:32])
    assert edge.size_kb(q) < edge.size_kb(path)
    assert (edge.onnx_predict(q, te["x"]).argmax(1) == p).mean() > 0.9
