from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch.nn as nn

from ibkd_seg.phase1.models import ResNet50CUB
from ibkd_seg.phase1.run_cub_r50_teacher_full import (
    EXPECTED_CONFIG_SHA256,
    _validate_config,
)
from ibkd_seg.phase1.train_timing import file_sha256


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "phase1/phase1_cub/configs/cub200_r50_224_b128_full_v3.json"
SCRIPT = ROOT / "phase1/phase1_cub/scripts/run_r50_224_teacher_full.sh"


class _BackboneStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2048, 1000)


class Phase1CubR50TeacherFullTest(unittest.TestCase):
    def test_locked_v3_config_is_accepted_byte_for_byte(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(CONFIG), EXPECTED_CONFIG_SHA256)
        _validate_config(config, config_path=CONFIG)

    def test_any_config_file_drift_is_rejected(self) -> None:
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        config["classification"]["teacher"]["external_pretraining"] = True
        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            changed.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "config_sha256|teacher"):
                _validate_config(config, config_path=changed)

    def test_resnet50_constructor_explicitly_disables_pretrained_weights(self) -> None:
        with patch("torchvision.models.resnet50", return_value=_BackboneStub()) as factory:
            model = ResNet50CUB(num_classes=200)
        factory.assert_called_once_with(weights=None)
        self.assertEqual(model.backbone.fc.out_features, 200)

    def test_h200_script_uses_only_the_v3_teacher_entrypoint(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("ibkd_seg.phase1.run_cub_r50_teacher_full", script)
        self.assertIn("cub200_r50_224_b128_full_v3.json", script)
        self.assertIn("--full-teacher", script)
        self.assertIn("--device cuda", script)
        self.assertNotIn("ibkd_seg.phase1.train_full", script)
        self.assertTrue(SCRIPT.stat().st_mode & 0o111)


if __name__ == "__main__":
    unittest.main()
