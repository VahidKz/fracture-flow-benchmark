import json
from xml.etree import ElementTree

import numpy as np

from fracture_flow_benchmark.cli import main


def test_one_command_produces_verified_data_and_both_figure_formats(tmp_path, capsys):
    output = tmp_path / "reproduction"
    assert main(["reproduce", "--output", str(output)]) == 0
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["acceptance"]["passed"]
    assert all(metrics["acceptance"]["checks"].values())
    assert (output / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in (output / "summary.svg").read_text()
    assert ElementTree.parse(output / "summary.svg").getroot().tag.endswith("svg")
    assert len(list(output.glob("*.csv"))) == 8
    profile = np.genfromtxt(output / "fracture_flux.csv", delimiter=",", names=True)
    np.testing.assert_allclose(np.mean(profile["q_m2_s"]), metrics["fracture"]["numerical_q_m2_s"])
    assert "Acceptance: PASS" in capsys.readouterr().out


def test_filesystem_failure_returns_a_clear_nonzero_status(tmp_path, capsys):
    path = tmp_path / "existing-file"
    path.write_text("keep me")
    assert main(["reproduce", "--output", str(path)]) == 2
    assert "Error:" in capsys.readouterr().err
    assert path.read_text() == "keep me"


def test_failed_accuracy_gate_returns_nonzero(tmp_path, monkeypatch, capsys):
    import fracture_flow_benchmark.experiments as experiments

    metrics = experiments.reproduce(tmp_path / "valid-run")
    metrics["acceptance"]["passed"] = False
    monkeypatch.setattr(experiments, "reproduce", lambda output: metrics)
    assert main(["reproduce", "--output", str(tmp_path)]) == 1
    assert "Acceptance: FAIL" in capsys.readouterr().out
