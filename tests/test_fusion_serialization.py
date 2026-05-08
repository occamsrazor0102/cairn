import pytest


def test_save_and_load_round_trips_with_metadata(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    import numpy as np
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import save_model, load_model

    X = [{"a": float(i % 5), "b": 1.0} for i in range(40)]
    y = [i % 2 for i in range(40)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "ebm.skops"
    save_model(model, out, metadata={"dataset_hash": "abc123", "n_samples": 40})
    loaded, metadata = load_model(out)
    assert metadata["dataset_hash"] == "abc123"
    Xt = np.array([[1.0, 1.0]])
    np.testing.assert_allclose(model.predict_proba(Xt), loaded.predict_proba(Xt), atol=1e-6)


def test_load_detects_tampering(tmp_path):
    pytest.importorskip("interpret")
    pytest.importorskip("skops")
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import save_model, load_model
    X = [{"a": float(i), "b": 1.0} for i in range(20)]
    y = [i % 2 for i in range(20)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "ebm.skops"
    save_model(model, out, metadata={"version": "v7"})
    sidecar = out.with_suffix(".skops.hmac")
    sidecar.write_text("0" * 64)
    with pytest.raises(ValueError, match="HMAC mismatch"):
        load_model(out, verify_hmac=True)


def test_legacy_pickle_load_warns_then_works(tmp_path, monkeypatch):
    pytest.importorskip("interpret")
    import pickle
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import load_model
    monkeypatch.setenv("CAIRN_ALLOW_LEGACY_PICKLE", "1")
    X = [{"a": float(i), "b": 1.0} for i in range(20)]
    y = [i % 2 for i in range(20)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "legacy.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    with pytest.warns(DeprecationWarning, match="legacy pickle"):
        loaded, metadata = load_model(out, verify_hmac=False)
    assert loaded is not None


def test_legacy_pickle_load_refuses_without_opt_in(tmp_path, monkeypatch):
    """Legacy pickle path must refuse without CAIRN_ALLOW_LEGACY_PICKLE=1 set."""
    pytest.importorskip("interpret")
    import pickle
    from cairn.fusion.training import train_ebm
    from cairn.fusion.serialization import load_model
    monkeypatch.delenv("CAIRN_ALLOW_LEGACY_PICKLE", raising=False)
    X = [{"a": float(i), "b": 1.0} for i in range(20)]
    y = [i % 2 for i in range(20)]
    model = train_ebm(X, y, feature_names=["a", "b"])
    out = tmp_path / "legacy.pkl"
    with open(out, "wb") as f:
        pickle.dump(model, f)
    with pytest.raises(RuntimeError, match="CAIRN_ALLOW_LEGACY_PICKLE"):
        load_model(out, verify_hmac=False)
