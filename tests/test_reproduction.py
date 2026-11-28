"""Regression tests use generated images, never manuscript training results."""
import json
import os
import sys
import unittest
import uuid
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch import nn

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
from srnet import engine
from srnet.artifacts import experiment_identity
from srnet.config import VARIANTS, load_config, resolve_variant
from srnet.data import ImageRecord, dataset_fingerprint, load_records, records_fingerprint
from srnet.inference import checkpoint_partition
from srnet.models import build_model
from srnet.runtime import load_checkpoint, save_checkpoint
from srnet.splits import prepare_splits
import bind_clinical
import plot_results
import style_baseline


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(4, 2)

    def forward(self, images, style_images=None):
        logits = self.classifier(self.dropout(self.conv(images).mean((2, 3))))
        return dict(logits=logits, scaled_logits=logits, temperature=logits.new_ones((len(images), 1)))


class ReproductionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.work = Path(os.environ.get("SRNET_TEST_ROOT", CODE.parent / "build/srnet-regression")) / uuid.uuid4().hex[:8]
        cls.work.mkdir(parents=True)
        cls.records = []
        rng = np.random.default_rng(91)
        for i in range(40):
            label = i % 2
            pixels = rng.integers(0, 80, (36, 32, 3), dtype=np.uint8)
            pixels[:, :, label] += 100
            name = f"generated_{i}.png"
            Image.fromarray(pixels).save(cls.work / name)
            cls.records.append(ImageRecord(f"generated_{i}", f"test_patient_{i // 2}", name, label))
        cls.manifest = cls.work / "manifest.csv"
        pd.DataFrame([r.__dict__ for r in cls.records]).to_csv(cls.manifest, index=False)
        print(f"Generated test artifacts: {cls.work}", flush=True)

    def config(self, name):
        config = load_config(CODE / "configs/public.yaml")
        config["experiment"].update(seeds=[73421, 18013], reference_seed=73421, split_mode="generate",
                                    split_file=str(self.work / "generated_splits.csv"), output=str(self.work / name))
        config["data"].update(name="generated", source=str(self.work), manifest=str(self.manifest), archive_sha256=None, image_size=32)
        config["training"].update(epochs=2, batch_size=8, num_workers=0, device="cpu")
        config["model"].update(pretrained=False, pretrained_path=None)
        return resolve_variant(config, "full")

    def fit(self, name):
        config = self.config(name)
        assignments = prepare_splits(self.records, config)
        folder = self.work / name / "fold"
        with patch.object(engine, "build_model", side_effect=lambda *a, **kw: TinyNetwork()):
            frame, metrics = engine.fit_fold(self.records, assignments, config, 73421, 1, folder)
        return config, assignments, folder, frame, metrics

    def test_paper_seeds_and_all_retained_folds(self):
        config = load_config(CODE / "configs/public.yaml")
        expected = [73421, 18013, 92051, 46807, 31573]
        self.assertEqual(config["experiment"]["seeds"], expected)
        self.assertEqual(config["experiment"]["reference_seed"], 73421)
        self.assertEqual(config["experiment"]["split_mode"], "replay")
        public = pd.read_csv(CODE / "protocol/public_manifest.csv", keep_default_na=False)
        clinical = pd.read_csv(CODE / "protocol/clinical_manifest_template.csv", keep_default_na=False)
        self.assertEqual(len(public), 204)
        self.assertEqual(public.label.sum(), 103)
        self.assertEqual(len(clinical), 576)
        self.assertEqual(clinical.patient_id.nunique(), 192)
        for cohort, manifest in [("public", public), ("clinical", clinical)]:
            records = [ImageRecord(r.sample_id, r.patient_id, r.path, r.label) for r in manifest.itertuples()]
            settings = load_config(CODE / f"configs/{cohort}.yaml")
            splits = prepare_splits(records, settings)
            self.assertEqual(len(splits), len(records) * 5)
            if cohort == "public":
                sizes = splits[splits.seed.eq(73421)].groupby("fold").size().tolist()
                self.assertEqual(sizes, [40, 41, 40, 42, 41])
            else:
                self.assertTrue(splits.groupby(["seed", "patient_id"]).fold.nunique().eq(1).all())

    def test_missing_replay_splits_fail_without_generation(self):
        config = self.config("missing_split")
        config["experiment"].update(split_mode="replay", split_file=str(self.work / "must_not_generate.csv"))
        with self.assertRaisesRegex(FileNotFoundError, "Retained split"):
            prepare_splits(self.records, config)
        self.assertFalse(Path(config["experiment"]["split_file"]).exists())

    def test_separate_paper_analyses_do_not_silently_use_main_splits(self):
        main = load_config(CODE / "configs/public.yaml")
        with self.assertRaisesRegex(ValueError, "appearance"):
            resolve_variant(main, "macenko")
        clinical = load_config(CODE / "configs/clinical.yaml")
        with self.assertRaisesRegex(ValueError, "scan"):
            resolve_variant(clinical, "raster")
        appearance = load_config(CODE / "configs/public_appearance.yaml")
        self.assertNotEqual(appearance["experiment"]["split_file"], main["experiment"]["split_file"])

    def test_partitioned_folds_and_seeds_accumulate(self):
        config = self.config("partitioned")
        with patch.object(engine, "build_model", side_effect=lambda *a, **kw: TinyNetwork()):
            output = engine.run_cross_validation(self.records, config, seeds=[73421], folds=[1, 2])
            first = pd.read_csv(output / "oof_predictions.csv")
            self.assertEqual(set(first.fold), {1, 2})
            engine.run_cross_validation(self.records, config, seeds=[73421], folds=[3, 4, 5], resume=True)
            first_seed = pd.read_csv(output / "oof_predictions.csv")
            self.assertEqual(len(first_seed), 40)
            self.assertEqual(set(first_seed.fold), {1, 2, 3, 4, 5})
            self.assertEqual(json.loads((output / "experiment.json").read_text())["status"], "partial")
            engine.run_cross_validation(self.records, config, seeds=[18013], resume=True)
            all_seeds = pd.read_csv(output / "oof_predictions.csv")
            self.assertEqual(len(all_seeds), 80)
            self.assertFalse(all_seeds.duplicated(["seed", "sample_id"]).any())
            self.assertEqual(pd.read_csv(output / "metrics_across_seeds.csv").seeds.iloc[0], 2)
            self.assertEqual(json.loads((output / "experiment.json").read_text())["status"], "complete")
            before = (output / "oof_predictions.csv").read_bytes()
            engine.run_cross_validation(self.records, config, seeds=[73421], folds=[1], resume=True)
            self.assertEqual(before, (output / "oof_predictions.csv").read_bytes())

    def test_modified_predictions_and_checkpoint_are_rejected(self):
        config, assignments, folder, _, _ = self.fit("cache")
        names = ["run.json", "best.pt", "validation_predictions.csv", "test_predictions.csv"]
        originals = {name: (folder / name).read_bytes() for name in names}
        for changed in ("scores", "checkpoint", "stored_metric", "missing_hashes"):
            with self.subTest(changed=changed):
                for name, content in originals.items():
                    (folder / name).write_bytes(content)
                if changed == "scores":
                    frame = pd.read_csv(folder / "test_predictions.csv")
                    frame["score"] = 1 - frame["score"]
                    frame.to_csv(folder / "test_predictions.csv", index=False)
                elif changed == "checkpoint":
                    with (folder / "best.pt").open("ab") as stream:
                        stream.write(b"audit corruption")
                else:
                    metadata = json.loads((folder / "run.json").read_text())
                    if changed == "stored_metric":
                        metadata["test_metrics"]["accuracy"] += 0.123
                    else:
                        metadata.pop("files_sha256")
                    (folder / "run.json").write_text(json.dumps(metadata))
                with self.assertRaises(ValueError):
                    engine.fit_fold(self.records, assignments, config, 73421, 1, folder, resume=True)

    def test_actual_image_change_is_detected_at_resume_and_evaluation(self):
        config, assignments, folder, _, _ = self.fit("image_change")
        checkpoint = load_checkpoint(folder / "best.pt")
        before = dataset_fingerprint(self.records, config["data"])
        records_hash = records_fingerprint(self.records)
        image = self.work / self.records[0].path
        old = image.read_bytes()
        try:
            Image.new("RGB", (32, 36), (5, 6, 7)).save(image)
            self.assertEqual(records_hash, records_fingerprint(self.records))
            self.assertNotEqual(before, dataset_fingerprint(self.records, config["data"]))
            with self.assertRaisesRegex(ValueError, "different data"):
                engine.fit_fold(self.records, assignments, config, 73421, 1, folder, resume=True)
            with self.assertRaisesRegex(ValueError, "actual image content"):
                checkpoint_partition(self.records, checkpoint, data=config["data"])
        finally:
            image.write_bytes(old)

    def test_conflicting_config_does_not_overwrite_existing_metadata(self):
        config = self.config("conflict")
        with patch.object(engine, "build_model", side_effect=lambda *a, **kw: TinyNetwork()):
            output = engine.run_cross_validation(self.records, config, seeds=[73421], folds=[1])
        before = {name: (output / name).read_bytes() for name in ("configuration.json", "experiment.json", "oof_predictions.csv")}
        changed = deepcopy(config)
        changed["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "different or legacy"):
            engine.run_cross_validation(self.records, changed, seeds=[73421], folds=[1], resume=True)
        for name, content in before.items():
            self.assertEqual(content, (output / name).read_bytes())

    def test_interrupted_resume_is_exact(self):
        config = self.config("resume_exact")
        assignments = prepare_splits(self.records, config)
        normal, interrupted = self.work / "normal", self.work / "interrupted"
        with patch.object(engine, "build_model", side_effect=lambda *a, **kw: TinyNetwork()):
            expected, _ = engine.fit_fold(self.records, assignments, config, 73421, 1, normal)

            def interrupt(filename, checkpoint):
                save_checkpoint(filename, checkpoint)
                if Path(filename).name == "last.pt" and checkpoint["epoch"] == 0:
                    raise RuntimeError("intentional test interruption")

            with patch.object(engine, "save_checkpoint", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "intentional test"):
                    engine.fit_fold(self.records, assignments, config, 73421, 1, interrupted)
            actual, _ = engine.fit_fold(self.records, assignments, config, 73421, 1, interrupted, resume=True)
        np.testing.assert_array_equal(expected.score, actual.score)
        first, second = load_checkpoint(normal / "last.pt"), load_checkpoint(interrupted / "last.pt")
        for key in first["model_state"]:
            self.assertTrue(torch.equal(first["model_state"][key], second["model_state"][key]))

    def test_histories_preserve_model_and_dataset_and_filter_seed(self):
        paths = []
        for model, seed in (("full", 73421), ("backbone", 73421), ("full", 18013)):
            path = self.work / "histories" / model / f"seed_{seed}/fold_1/history.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(dict(dataset=["generated"] * 2, model=[model] * 2, seed=[seed] * 2, fold=[1] * 2,
                              epoch=[1, 2], train_loss=[0.8, 0.7], validation_loss=[0.9, 0.8], validation_auroc=[0.5, 0.6])).to_csv(path, index=False)
            paths.append(path)
        captured = []

        def capture(figure, destination):
            captured.append(destination)
            plot_results.plt.close(figure)

        with patch.object(plot_results, "save_figure", side_effect=capture):
            plot_results.plot_histories(paths, self.work / "plots", 73421)
        self.assertEqual(len(captured), 2)
        self.assertEqual(len(set(captured)), 2)
        self.assertIn("full", captured[0].parts)
        self.assertIn("backbone", captured[1].parts)
        with patch.object(plot_results, "save_figure") as saver:
            with self.assertRaisesRegex(ValueError, "overwrite"):
                plot_results.plot_histories([paths[0], paths[0]], self.work / "plots")
            saver.assert_not_called()

    def test_rgb_baseline_partitioned_runs_accumulate(self):
        config = self.config("rgb")
        filename = self.work / "rgb.yaml"
        filename.write_text(yaml.safe_dump(config), encoding="utf-8")
        output = self.work / "rgb_output"
        for folds in ([1, 2], [3, 4, 5]):
            argv = ["style_baseline.py", "--config", str(filename), "--output", str(output), "--seeds", "73421",
                    "--folds", *map(str, folds), "--resume"]
            with patch.object(sys, "argv", argv):
                style_baseline.main()
        frame = pd.read_csv(output / "oof_predictions.csv")
        self.assertEqual(len(frame), 40)
        self.assertEqual(set(frame.fold), set(range(1, 6)))

    def test_clinical_binding_preserves_real_ids_and_leaves_patient_reference_empty(self):
        template = self.work / "binding_template.csv"
        mapping = self.work / "binding_paths.csv"
        pd.DataFrame([dict(sample_id="real_retained_id", patient_id="retained_patient", path="", label=1, patient_label="")]).to_csv(template, index=False)
        pd.DataFrame([dict(sample_id="real_retained_id", path=self.records[0].path)]).to_csv(mapping, index=False)
        output = self.work / "bound_clinical.csv"
        bind_clinical.bind_images(template, mapping, self.work, output)
        records = load_records(dict(source=str(self.work), manifest=str(output), kind="clinical"))
        self.assertEqual(records[0].patient_id, "retained_patient")
        self.assertEqual(records[0].label, 1)
        self.assertEqual(records[0].patient_label, -1)

    def test_all_model_variants_forward_and_full_backward(self):
        import gc

        for variant in VARIANTS:
            config = self.config("model_shapes")
            config["experiment"]["analysis"] = "new_experiment"
            config["data"]["image_size"] = 224
            config = resolve_variant(config, variant)
            model = build_model(config, pretrained=False)
            x = torch.rand(2, 3, 224, 224)
            if variant == "full":
                result = model(x)
                result["scaled_logits"].square().mean().backward()
                for part in (model.adapter, model.routing.forward_scan, model.routing.backward_scan, model.head.temperature_branch):
                    self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in part.parameters()))
            else:
                model.eval()
                with torch.no_grad():
                    result = model(x)
            self.assertEqual(tuple(result["scaled_logits"].shape), (2, 2))
            self.assertTrue(torch.isfinite(result["scaled_logits"]).all())
            del model, x, result
            gc.collect()


if __name__ == "__main__":
    unittest.main(verbosity=2)
