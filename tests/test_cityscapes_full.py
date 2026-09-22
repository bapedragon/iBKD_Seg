import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ibkd_seg.cityscapes.full_checkpoint import save_checkpoint, load_checkpoint, save_best, verify_artifact
from ibkd_seg.cityscapes.full_data import epoch_order, sample_seed, train_loader
from ibkd_seg.cityscapes.official_full import validation_due, improves, validate_full_config


class IndexedData(Dataset):
    config = dict(seed=1, batch_size=8, data_workers=0)

    def __len__(self):
        return 2975

    def __getitem__(self, index):
        return index


class FullTrainingContracts(unittest.TestCase):
    def test_resume_exact_remaining_samples_and_last_batch(self):
        data = IndexedData()
        device = torch.device("cpu")
        torch.manual_seed(1)
        rng = torch.get_rng_state().clone()
        original = list(train_loader(data, 7, 0, device))
        remaining = list(train_loader(data, 7, 123, device))
        torch.testing.assert_close(rng, torch.get_rng_state(), rtol=0, atol=0)
        self.assertEqual(len(original), 372)
        self.assertEqual(len(original[-1]), 7)
        self.assertEqual(torch.cat(original).unique().numel(), 2975)
        torch.testing.assert_close(torch.cat(original[123:]), torch.cat(remaining), rtol=0, atol=0)
        self.assertEqual(torch.cat(original).tolist(), epoch_order(2975, 1, 7))
        self.assertNotEqual(epoch_order(2975, 1, 7), epoch_order(2975, 1, 8))
        self.assertNotEqual(sample_seed(1, 7, "a"), sample_seed(1, 8, "a"))

    def test_upstream_zero_based_validation_and_accuracy_selection(self):
        config = dict(epochs=216, validation_every=4)
        epochs = [e for e in range(1, 217) if validation_due(e, config)]
        self.assertEqual(epochs, list(range(1, 217, 4)) + [216])
        best = dict(metrics=dict(pixel_accuracy=0.8, miou=0.4))
        self.assertFalse(improves(dict(pixel_accuracy=0.8, miou=0.7), best))
        self.assertFalse(improves(dict(pixel_accuracy=0.7, miou=0.9), best))
        self.assertTrue(improves(dict(pixel_accuracy=0.81, miou=0.3), best))

    def test_crop512_final_80000_protocol_is_locked(self):
        root = Path(__file__).resolve().parents[1]
        path = root / (
            "phase4/phase4_cityscapes/configs/"
            "paper_l16_crop512_final80000_v19.json"
        )
        config = json.loads(path.read_text())
        validate_full_config(config)
        self.assertEqual(config["total_steps"], 80000)
        self.assertEqual(config["crop_size"], 512)
        self.assertEqual(config["decoder_layers"], 1)
        self.assertEqual(config["guidance_beta_by_method"], {
            "lg": 0.05,
            "alg": 0.05,
            "ibkd": 0.5,
        })
        self.assertEqual(config["ibkd_fusion_ratio"], 0.25)
        self.assertTrue(config["ibkd_deterministic_candidate"])
        self.assertTrue(validation_due(216, config, final=True))

        changed = copy.deepcopy(config)
        changed["guidance_beta_by_method"]["ibkd"] = 0.25
        with self.assertRaisesRegex(ValueError, "protocol mismatch"):
            validate_full_config(changed)

    def test_existing_crop768_full_protocol_remains_accepted(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((
            root / "phase4/phase4_cityscapes/configs/official_l16_full_v1.json"
        ).read_text())
        validate_full_config(config)

    def test_atomic_generations_fallback_portable_best_and_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "run"
            root.mkdir()
            identity = dict(method="ibkd", seed=1)
            model = torch.nn.Linear(2, 1)
            score = dict(pixel_accuracy=0.8, miou=0.4)
            best = save_best(root, model, identity, 1, score)
            payload = dict(signature=identity, progress=dict(global_step=100, best=best), marker="first")
            first = save_checkpoint(root, payload)
            next_best = save_best(root, model, identity, 5, dict(pixel_accuracy=0.9, miou=0.3))
            payload = copy.deepcopy(payload)
            payload.update(marker="second", progress=dict(global_step=200, best=next_best))
            second = save_checkpoint(root, payload)
            verify_artifact(root, first["current"]["best"])
            verify_artifact(root, second["current"]["best"])
            with self.assertRaisesRegex(ValueError, "identity differs"):
                load_checkpoint(root / "resume.json", dict(method="lg", seed=1), root)
            (root / second["current"]["file"]).write_bytes(b"interrupted-copy")
            restored, info = load_checkpoint(root / "resume.json", identity, Path(folder) / "new")
            self.assertEqual(restored["marker"], "first")
            self.assertEqual(info["generation"], "previous")
            self.assertEqual(len(info["fallback_errors"]), 1)
            verify_artifact(Path(folder) / "new", best)
            # A later checkpoint must keep its own and the previous best; older
            # unreferenced weights may be removed without breaking either run.
            third = save_checkpoint(root, restored)
            verify_artifact(root, third["current"])
            verify_artifact(root, third["current"]["best"])

    def test_checkpoint_pointer_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            record = dict(file="../external.pt", bytes=0, sha256="bad", best=None)
            (root / "resume.json").write_text(json.dumps(dict(format=1, current=record)))
            with self.assertRaisesRegex(RuntimeError, "Unsafe checkpoint"):
                load_checkpoint(root / "resume.json", {}, root)


if __name__ == "__main__":
    unittest.main()
