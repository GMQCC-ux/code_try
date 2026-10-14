"""Simulate power loss on synthetic data, without touching experiment outputs."""
import os
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch

import data
import train
import xrun
from resume_state import (
    atomic_csv_save, atomic_torch_save, atomic_write, read_epoch_state,
    read_json, restore_epoch_state, save_epoch_state,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.BatchNorm2d(4), torch.nn.ReLU(), torch.nn.Dropout2d(0.2),
        )
        self.seg = torch.nn.Conv2d(4, 1, 1)
        self.edge = torch.nn.Conv2d(4, 1, 1)

    def forward(self, x):
        features = self.features(x)
        seg, edge = self.seg(features), self.edge(features)
        return seg, [seg * 0.9, seg * 1.1, seg * 0.8], [edge, edge * 0.9, edge * 1.1]


def assert_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for name in left:
            assert_equal(left[name], right[name])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for first, second in zip(left, right):
            assert_equal(first, second)
    else:
        assert left == right, (left, right)


def make_dataset(root):
    generator = np.random.default_rng(42)
    for folder in ["images", "masks"]:
        (root / folder).mkdir(parents=True)
    for index in range(6):
        image = generator.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        mask = (image[:, :, 0] > 127).astype(np.uint8) * 255
        assert cv2.imwrite(str(root / "images" / f"{index}.png"), image)
        assert cv2.imwrite(str(root / "masks" / f"{index}.png"), mask)


def run_training(root, dataset_root, interrupt_epoch=None):
    root.mkdir(parents=True, exist_ok=True)
    configuration = SimpleNamespace(**{
        name: value for name, value in vars(train.config).items() if not name.startswith("__")
    })
    configuration.batch_size = 2
    configuration.num_epochs = 3
    configuration.main_dataset_dir = str(dataset_root) + "/"
    configuration.train_images_dir = "images/*.png"
    configuration.train_masks_dir = "masks/*.png"
    overrides = {
        "config": configuration, "auto_epoch_resume": True,
        "main_dataset_dir": configuration.main_dataset_dir,
        "train_images_dir": configuration.train_images_dir,
        "train_masks_dir": configuration.train_masks_dir,
        "batch_size": 2, "num_epochs": 3,
        "train_augmentation_online": True, "include_original_train": False,
        "use_amp": False, "use_lr_schedule": True,
        "resume_state_file": str(root / "resume" / "latest.pt"),
        "checkpoint_dir": str(root / "checkpoints"),
        "train_log_loss_file": str(root / "train-loss.csv"),
        "build_model": lambda pretrained=True: TinyModel(),
        "get_loss_fn": lambda: torch.nn.BCEWithLogitsLoss(),
        "get_edge_loss_fn": lambda: torch.nn.BCEWithLogitsLoss(),
        "prepare_train_dirs": lambda: (root / "checkpoints").mkdir(exist_ok=True),
        "append_run_log": lambda *args: None,
        "sanity_log": lambda *args: None, "save_sample": lambda *args: None,
        "print_stats": lambda *args: None, "all_stats": lambda *args: None,
    }
    original_epoch = train.train_one_epoch

    def run_or_interrupt(model, loader, optimizer, loss_fn, edge_loss_fn, device, epoch, scaler):
        if epoch == interrupt_epoch:
            # Update a real partial epoch, then lose all its in-memory progress.
            model.train()
            images, target = next(iter(loader))
            optimizer.zero_grad()
            prediction = model(images)[0]
            loss_fn(prediction, target).backward()
            optimizer.step()
            random.random()
            np.random.rand()
            torch.rand(7)
            raise RuntimeError("SIMULATED_POWER_LOSS")
        return original_epoch(model, loader, optimizer, loss_fn, edge_loss_fn, device, epoch, scaler)

    overrides["train_one_epoch"] = run_or_interrupt
    with patch.multiple(train, **overrides), patch.object(data, "size", (16, 16)), \
            patch.dict(os.environ, {"XRUN_SEED": "56"}), \
            patch.object(torch.cuda, "is_available", return_value=False), \
            patch.object(torch.cuda, "device_count", return_value=0):
        try:
            train.main()
        except RuntimeError as error:
            if str(error) != "SIMULATED_POWER_LOSS":
                raise
            assert interrupt_epoch is not None
    return read_epoch_state(root / "resume" / "latest.pt")


def verify_training(root):
    dataset = root / "data"
    make_dataset(dataset)
    continuous = run_training(root / "continuous", dataset)
    for interrupted_epoch in [1, 2]:
        output = root / f"interrupted-{interrupted_epoch}"
        partial = run_training(output, dataset, interrupt_epoch=interrupted_epoch)
        assert partial["epoch"] == interrupted_epoch - 1
        resumed = run_training(output, dataset)
        assert resumed["epoch"] == continuous["epoch"] == 3
        for name in ["model_state_dict", "optimizer_state_dict", "scheduler_state_dict", "rng_state"]:
            assert_equal(continuous[name], resumed[name])
        assert [row[2] for row in continuous["history"]] == [row[2] for row in resumed["history"]]
        assert len(resumed["history"]) == 3
        print(f"PASS: interruption inside epoch {interrupted_epoch}; exact CPU weights/optimizer/LR/RNG/loss replay")
        # Re-running a completed training must not append another three epochs.
        completed_again = run_training(output, dataset)
        assert completed_again["epoch"] == 3 and len(completed_again["history"]) == 3
    print("PASS: completed training is not repeated; num_epochs is a total target")

    checkpoint = root / "continuous" / "resume" / "latest.pt"
    original = read_epoch_state(checkpoint)

    def unfinished_write(stream):
        stream.write(b"incomplete")
        raise RuntimeError("SIMULATED_SAVE_FAILURE")

    try:
        atomic_write(checkpoint, unfinished_write, binary=True)
    except RuntimeError:
        pass
    assert_equal(read_epoch_state(checkpoint), original)
    print("PASS: interruption while saving leaves the previous complete commit intact")
    atomic_write(checkpoint, lambda stream: stream.write(b"invalid"), binary=True)
    fallback = read_epoch_state(checkpoint)
    assert fallback["epoch"] == 2
    print("PASS: damaged latest state falls back to preceding complete epoch")

    try:
        restore_epoch_state(original, None, None, None, None, None, "changed-config", 3)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("Changed configuration was accepted")
    print("PASS: changed configuration is rejected")


def verify_scaler(root):
    network = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(network.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3)
    scaler = torch.amp.GradScaler("cpu")
    generator = torch.Generator().manual_seed(42)
    loss = network(torch.randn(2, 3)).square().mean()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    scheduler.step(1)
    path = root / "amp.pt"
    save_epoch_state(network, optimizer, scheduler, scaler, generator, 1, [], 3, "amp", path)
    original = read_epoch_state(path)
    recovered = torch.nn.Linear(3, 1)
    recovered_optimizer = torch.optim.AdamW(recovered.parameters(), lr=0.5)
    recovered_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(recovered_optimizer, T_0=3)
    recovered_scaler = torch.amp.GradScaler("cpu", init_scale=128)
    restore_epoch_state(original, recovered, recovered_optimizer, recovered_scheduler,
                        recovered_scaler, torch.Generator(), "amp", 3)
    assert_equal(scaler.state_dict(), recovered_scaler.state_dict())
    assert_equal(optimizer.state_dict(), recovered_optimizer.state_dict())
    assert_equal(scheduler.state_dict(), recovered_scheduler.state_dict())
    print("PASS: AMP scaler, optimizer, and scheduler round-trip")


def verify_pipeline(root):
    root.mkdir()
    checkpoint_dir = root / "checkpoints"
    validation = root / "val"
    testing = root / "test"
    events = []
    interrupted = [False]

    def fake_script(name):
        events.append(name)
        if name == "train.py":
            atomic_torch_save({"model_state_dict": {}}, checkpoint_dir / "best.pth")
        elif name == "val_eval.py" and not interrupted[0]:
            interrupted[0] = True
            raise RuntimeError("SIMULATED_POWER_LOSS")
        elif name == "check_val.py":
            atomic_csv_save(validation / "best_val_weights.csv", ["weight_file", "dice_score"], [["best.pth", 0.9]])
        elif name == "test_eval.py":
            atomic_csv_save(testing / "pred-csv" / "summary_metrics.csv", ["Checkpoint", "Dice"], [["best.pth", 0.8]])

    overrides = {
        "tries": 1, "resume_train": 0, "resumed_chkpt": "", "output_root": str(root),
        "run_state_file": str(root / "resume" / "run_state.json"),
        "resume_state_file": str(root / "resume" / "latest.pt"),
        "checkpoint_dir": str(checkpoint_dir), "eval_root_val": str(validation),
        "eval_root_test": str(testing), "training_signature": lambda *args: ("test-signature", {}),
        "run_script": fake_script, "append_run_log": lambda *args: None,
    }
    with patch.multiple(xrun, **overrides), patch.object(
        xrun, "glob", side_effect=lambda pattern: [] if pattern.endswith("*.pth") else ["synthetic"]
    ):
        try:
            xrun.run_single_iteration_resumable()
        except RuntimeError as error:
            assert str(error) == "SIMULATED_POWER_LOSS"
        assert read_json(overrides["run_state_file"])["phase"] == "val"
        xrun.run_single_iteration_resumable()
        assert events == ["train.py", "val_eval.py", "val_eval.py", "check_val.py", "test_eval.py"]
        assert read_json(overrides["run_state_file"])["phase"] == "complete"
        xrun.run_single_iteration_resumable()
        assert events.count("train.py") == 1
    print("PASS: pipeline resumes validation after training, finishes test, and does not restart a completed run")


def main():
    torch.set_num_threads(2)
    Path(".verification").mkdir(exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="epoch-resume-", dir=".verification")).resolve()
    verify_training(root)
    verify_scaler(root)
    verify_pipeline(root / "pipeline")
    print(f"All checks passed. Synthetic artifacts retained at {root}")


if __name__ == "__main__":
    main()
