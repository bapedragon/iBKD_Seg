from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from ibkd_seg.phase1.models import ResNet50CUB
from ibkd_seg.phase1.run_cub_r50_teacher_full import (
    CHECKPOINT_PURPOSE,
    EXPECTED_CONFIG_SHA256,
    EXPECTED_VALIDATION_HASH,
    _validate_config,
    load_scientific_teacher,
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

    def test_shared_teacher_checkpoint_uses_safe_weights_only_load(self) -> None:
        model = nn.Linear(1, 1)
        state = model.state_dict()
        metadata = {
            "purpose": CHECKPOINT_PURPOSE,
            "scientific_result": True,
            "dataset": "CUB-200-2011",
            "num_classes": 200,
            "architecture": "torchvision_resnet50",
            "initialization": "scratch",
            "external_pretraining": False,
            "input_size": 224,
            "epochs": 200,
            "seed": 1,
            "batch_size": 128,
            "selection_metric": "validation_macro_top1",
            "selection_tie_break": "earlier_epoch",
            "validation_image_ids_sha256": EXPECTED_VALIDATION_HASH,
            "full_config_sha256": EXPECTED_CONFIG_SHA256,
            "official_test_evaluations_at_checkpoint_write": 0,
            "model_state_sha256": "state-hash",
        }
        incompatible = MagicMock(missing_keys=[], unexpected_keys=[])
        model.load_state_dict = MagicMock(return_value=incompatible)
        with (
            patch(
                "ibkd_seg.phase1.run_cub_r50_teacher_full.torch.load",
                return_value={"model": state, "metadata": metadata},
            ) as loader,
            patch(
                "ibkd_seg.phase1.run_cub_r50_teacher_full.ResNet50CUB",
                return_value=model,
            ),
            patch(
                "ibkd_seg.phase1.run_cub_r50_teacher_full.state_dict_sha256",
                return_value="state-hash",
            ),
            patch(
                "ibkd_seg.phase1.run_cub_r50_teacher_full.file_sha256",
                return_value="file-hash",
            ),
        ):
            loaded, _, checkpoint_hash, state_hash = load_scientific_teacher(
                Path("teacher.pt"), device=torch.device("cpu")
            )

        loader.assert_called_once_with(
            Path("teacher.pt"), map_location="cpu", weights_only=True
        )
        model.load_state_dict.assert_called_once_with(state, strict=True)
        self.assertIs(loaded, model)
        self.assertFalse(loaded.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in loaded.parameters()))
        self.assertEqual(checkpoint_hash, "file-hash")
        self.assertEqual(state_hash, "state-hash")


if __name__ == "__main__":
    unittest.main()
