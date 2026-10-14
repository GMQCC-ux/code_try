"""Durable epoch-boundary state, independent of training/evaluation imports."""
import csv
import hashlib
import json
import os
import pickle
import random
import shutil

import numpy as np
import torch


def atomic_write(path, writer, binary=False):
    """Commit only after the temporary file has been fully written and flushed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = str(path) + ".tmp"
    kwargs = {} if binary else {"encoding": "utf-8", "newline": ""}
    with open(temporary, "wb" if binary else "w", **kwargs) as stream:
        writer(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_torch_save(state, path):
    atomic_write(path, lambda stream: torch.save(state, stream), binary=True)


def atomic_json_save(state, path):
    atomic_write(path, lambda stream: json.dump(state, stream, indent=2, ensure_ascii=False))


def read_json(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def atomic_csv_save(path, header, rows):
    def write(stream):
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    atomic_write(path, write)


def upsert_csv_row(path, row, key_index=0):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    header, previous = rows[0], rows[1:]
    previous = [item for item in previous if item[key_index] != str(row[key_index])]
    atomic_csv_save(path, header, [*previous, row])


def capture_rng_state(loader_generator):
    np_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # Store plain values/tensors so weights_only=True can read the checkpoint.
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader": loader_generator.get_state(),
    }


def restore_rng_state(state, loader_generator):
    random.setstate(state["python"])
    name, keys, position, has_gauss, cached_gauss = state["numpy"]
    np.random.set_state((name, np.array(keys, dtype=np.uint32), position, has_gauss, cached_gauss))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state["torch_cuda"]:
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count changed; exact resume is not supported")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
    loader_generator.set_state(state["loader"].cpu())


def training_signature(config, images, masks):
    names = (
        "batch_size", "num_epochs", "seed_val", "random_seed", "H", "W",
        "num_Classes", "encoder_name_str", "eca_use_spatial_gradient",
        "lr", "use_lr_schedule", "cosine_T0", "cosine_T_mult", "eta_min",
        "use_amp", "train_augmentation_online", "albumb_aug", "complex_aug",
        "image_only_aug", "cutmix_augmnet", "include_original_train",
        "deep_supervision", "deep_supervision_weights", "normalize_deep_weights",
        "edge_supervision", "edge_weights", "normalize_edge_weights",
        "loss_name", "edge_loss_name", "main_dataset_dir",
        "train_images_dir", "train_masks_dir", "val_images_dir", "val_masks_dir",
        "test_images_dir", "test_masks_dir", "gpu_list",
    )
    settings = {name: getattr(config, name) for name in names}
    import albumentations
    import timm
    settings["runtime"] = {
        "torch": str(torch.__version__), "numpy": np.__version__,
        "albumentations": albumentations.__version__, "timm": timm.__version__,
        "cuda": torch.version.cuda,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "xrun_seed": int(os.environ.get("XRUN_SEED", config.seed_val)),
    }
    settings["source_hashes"] = {}
    for filename in ["model.py", "data.py", "loss.py", "train.py", "utils.py", "resume_state.py"]:
        with open(os.path.join(os.path.dirname(__file__), filename), "rb") as source:
            settings["source_hashes"][filename] = hashlib.sha256(source.read()).hexdigest()
    files = []
    for path in [*images, *masks]:
        stat = os.stat(path)
        files.append((os.path.abspath(path), stat.st_size, stat.st_mtime_ns))
    payload = {"settings": settings, "training_files": files}
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), settings


def save_epoch_state(model, optimizer, scheduler, scaler, loader_generator,
                     epoch, history, target_epochs, signature, path, settings=None):
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    state = {
        "format_version": 1,
        "epoch": epoch,
        "target_epochs": target_epochs,
        "signature": signature,
        "settings": settings,
        "model_state_dict": module.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(loader_generator),
        "history": history,
    }
    # Keep the preceding complete commit as a fallback for disk corruption.
    if os.path.isfile(path):
        with open(path, "rb") as previous:
            atomic_write(str(path) + ".previous", lambda stream: shutil.copyfileobj(previous, stream), binary=True)
    atomic_torch_save(state, path)
    return state


def read_epoch_state(path):
    if not os.path.isfile(path):
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, EOFError, RuntimeError, ValueError, pickle.UnpicklingError):
        backup = str(path) + ".previous"
        if not os.path.isfile(backup):
            raise
        print(f"[RESUME] Latest state unreadable; using preceding complete epoch: {backup}")
        state = torch.load(backup, map_location="cpu", weights_only=True)
        # Repair latest so the next save never backs up a corrupted file.
        with open(backup, "rb") as previous:
            atomic_write(path, lambda stream: shutil.copyfileobj(previous, stream), binary=True)
        return state


def restore_epoch_state(state, model, optimizer, scheduler, scaler, loader_generator,
                        signature, target_epochs):
    if state.get("format_version") != 1:
        raise RuntimeError("Unsupported recovery checkpoint format")
    if state["signature"] != signature or state["target_epochs"] != target_epochs:
        raise RuntimeError(
            "Training settings/data/runtime changed. Keep them unchanged to resume, "
            "or choose a different output_root for a new experiment."
        )
    module = model.module if isinstance(model, torch.nn.DataParallel) else model
    module.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    # Must be last: setup/model construction may have consumed random values.
    restore_rng_state(state["rng_state"], loader_generator)
    return state["epoch"], list(state["history"])


def sync_training_log(path, history):
    atomic_csv_save(path, ["checkpoint", "epoch", "loss", "time"], history)
