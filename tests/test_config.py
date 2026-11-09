from dataclasses import asdict
from pathlib import Path

import pytest

from neurocinema.config import Config, ConfigError, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_installation_toml_mirrors_defaults():
    assert asdict(load_config([ROOT / "config" / "installation.toml"])) == asdict(Config())


def test_later_files_override(tmp_path):
    a = tmp_path / "a.toml"
    a.write_text("[targets]\nleft_yaw_deg = -50\n")
    b = tmp_path / "b.toml"
    b.write_text("[targets]\nleft_yaw_deg = -45.5\n[gestures.nod]\npeak_dps = 40\n")
    cfg = load_config([a, b])
    assert cfg.targets.left_yaw_deg == -45.5
    assert cfg.gestures.nod.peak_dps == 40.0
    assert cfg.gestures.shake.peak_dps == 70.0


@pytest.mark.parametrize("text", [
    "[targets]\nleft_yaw = -40\n",                 # typo
    "[imu]\nforward_axis = 'q'\n",
    "[targets]\nleft_yaw_deg = 10\n",              # left must be left of middle
    "[osc]\ndestinations = ['localhost']\n",
    "[interaction]\nmin_dwell_s = 'long'\n",
    "[gestures.nod]\nfirst_sign = 2\n",
    "[imu]\nbackend = 'kalman'\n",
    "[imu]\nvqf_bias_from = 'magic'\n",
])
def test_bad_config_is_rejected(tmp_path, text):
    p = tmp_path / "bad.toml"
    p.write_text(text)
    with pytest.raises(ConfigError):
        load_config([p])
