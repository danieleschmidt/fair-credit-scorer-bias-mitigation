import json
import subprocess
import sys
from pathlib import Path


def test_creates_output_json(tmp_path):
    out_file = tmp_path / "metrics.json"
    data_file = tmp_path / "credit.csv"
    cmd = [
        sys.executable,
        "-m",
        "fair_credit_scorer_bias_mitigation.cli",
        "--method",
        "baseline",
        "--test-size",
        "0.5",
        "--data-path",
        str(data_file),
        "--output-json",
        str(out_file),
    ]
    subprocess.run(cmd, check=True)
    assert out_file.exists()
    loaded = json.loads(out_file.read_text())
    assert "accuracy" in loaded
    assert "overall" in loaded
    assert "by_group" in loaded
