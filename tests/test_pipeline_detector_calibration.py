import pytest
from pathlib import Path
from cairn.pipeline import CairnPipeline
from cairn.config import load_config
from cairn.calibration import DetectorCalibrator


def test_detector_calibration_path_loads_when_configured(tmp_path):
    cal = DetectorCalibrator()
    cal.fit("preamble", [0.1, 0.3, 0.5, 0.7, 0.9], [0, 0, 1, 1, 1])
    cal_file = tmp_path / "cal.json"
    cal.save(cal_file)

    cfg = load_config(Path(__file__).parent.parent / "configs" / "screening.yaml")
    cfg.setdefault("fusion", {})["detector_calibration_path"] = str(cal_file)
    p = CairnPipeline(cfg)

    assert p._detector_calibrator is not None, (
        "pipeline must load DetectorCalibrator when detector_calibration_path is set"
    )
    assert p._detector_calibrator.has("preamble")


def test_detector_calibration_path_none_when_not_configured():
    cfg = load_config(Path(__file__).parent.parent / "configs" / "screening.yaml")
    cfg.setdefault("fusion", {}).pop("detector_calibration_path", None)
    p = CairnPipeline(cfg)
    assert p._detector_calibrator is None


def test_calibrator_transforms_p_llm_in_layer_results(tmp_path, monkeypatch):
    cal = DetectorCalibrator()
    cal.fit("preamble", [0.0, 0.4, 0.6, 1.0], [0, 0, 1, 1])
    cal_file = tmp_path / "cal.json"
    cal.save(cal_file)

    cfg = load_config(Path(__file__).parent.parent / "configs" / "screening.yaml")
    cfg.setdefault("fusion", {})["detector_calibration_path"] = str(cal_file)
    p = CairnPipeline(cfg)

    observed = []
    orig_fuse = p._fusion.fuse

    def spy_fuse(results, **kwargs):
        for r in results:
            if r.layer_id == "preamble":
                observed.append({"p_llm": r.p_llm, "calibrator_applied": r.calibrator_applied})
        return orig_fuse(results, **kwargs)

    monkeypatch.setattr(p._fusion, "fuse", spy_fuse)
    p.analyze("Certainly! Here is your comprehensive answer. " * 5)

    assert observed, "preamble did not run"
    assert observed[0]["calibrator_applied"] is True, (
        "calibrator_applied flag must be set when calibrator transforms p_llm"
    )
