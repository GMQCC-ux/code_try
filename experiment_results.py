"""Append final validation/test metrics to the shared user workbook."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKBOOK = PROJECT_ROOT / "运行结果.xlsx"
SUPPORT = Path(__file__).resolve().parent / "results_support"
RUNTIME = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"


def _csv_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def _metric_row(path, checkpoint):
    matches = [row for row in _csv_rows(path) if row["Checkpoint"] == checkpoint]
    if matches:
        return matches[-1]
    return None


def _metrics(row):
    result = {}
    for metric in ["dice", "iou", "hd95", "precision", "recall"]:
        label = {"dice":"Dice", "iou":"IoU", "hd95":"HD95",
                 "precision":"Precision", "recall":"Recall"}[metric]
        key = label if label in row else metric + "_macro_avg"
        result[metric] = float(row[key])
    return result


def snapshot_validation(config):
    """Legacy xrun prunes metrics; retain the selected row before pruning."""
    root = Path(config.eval_root_val)
    best = _csv_rows(root / "best_val_weights.csv")[0]["weight_file"]
    row = _metric_row(root / "pred-csv/summary_metrics.csv", best)
    path = root / "selected_validation_metrics.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if row is not None:
        existing[best] = row
        temporary = path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as stream:
            json.dump(existing, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def build_request(config, match_existing=False):
    validation_root = Path(config.eval_root_val)
    best = _csv_rows(validation_root / "best_val_weights.csv")[0]["weight_file"]
    validation = _metric_row(validation_root / "pred-csv/summary_metrics.csv", best)
    if validation is None:
        # A locked legacy checkpoint is a copy of the timestamp-named weights.
        base_name = best.split("_best_iteration_")[0] + ".pth" if "_best_iteration_" in best else best
        snapshot = validation_root / "selected_validation_metrics.json"
        saved = json.loads(snapshot.read_text(encoding="utf-8")) if snapshot.exists() else {}
        validation = saved.get(best) or saved.get(base_name)
    test = _metric_row(Path(config.eval_root_test) / "pred-csv/summary_metrics.csv", best)
    if validation is None or test is None:
        raise RuntimeError("Final validation/test metrics missing for the selected checkpoint")
    checkpoint = str((Path(config.checkpoint_dir) / best).resolve())
    run_id = hashlib.sha256(checkpoint.encode("utf-8")).hexdigest()[:24]
    # Use recorded settings when available rather than post-training config edits.
    state_path = Path(getattr(config, "run_state_file", "output/resume/run_state.json"))
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    settings = state.get("settings", {})
    seed = settings.get("runtime", {}).get("xrun_seed")
    if seed is None:
        seed = int(os.environ.get("XRUN_SEED", config.seed_val + (1 if config.random_seed else 0)))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    finished = datetime.fromtimestamp(
        (Path(config.eval_root_test) / "pred-csv/summary_metrics.csv").stat().st_mtime,
        ZoneInfo("Asia/Shanghai"),
    )
    return {
        "workbook":str(WORKBOOK),
        "dataset":Path(settings.get("main_dataset_dir",config.main_dataset_dir).rstrip("/\\")).name,
        "epochs":int(settings.get("num_epochs",config.num_epochs)),
        "batchSize":int(settings.get("batch_size",config.batch_size)), "seed":seed,
        "checkpoint":best,"runId":run_id,
        "finishedAt":finished.strftime("%Y-%m-%d %H:%M:%S"),
        "validation":_metrics(validation),"test":_metrics(test),
        "matchExisting":match_existing,
    }


def write_request(request):
    # A persistent lock serialises separate experiments sharing this workbook.
    import msvcrt
    lock_path = SUPPORT / "workbook.lock"
    SUPPORT.mkdir(exist_ok=True)
    with open(lock_path, "a+b") as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_LOCK,1)
        try:
            if not WORKBOOK.exists():
                raise FileNotFoundError(f"Workbook not found: {WORKBOOK}")
            if (WORKBOOK.parent / ("~$" + WORKBOOK.name)).exists():
                raise PermissionError("Close 运行结果.xlsx in Excel, then re-run to save results")
            backup = SUPPORT / "运行结果.before-auto-write.bak"
            if not backup.exists():
                shutil.copy2(WORKBOOK,backup)
            request_path = SUPPORT / "pending_results.json"
            request_path.write_text(json.dumps(request,ensure_ascii=False),encoding="utf-8")
            node = RUNTIME / "node/bin/node.exe"
            if not node.exists():
                raise RuntimeError("Bundled spreadsheet runtime missing; results remain in their CSV files")
            process = subprocess.run([str(node),str(SUPPORT / "update_results.mjs"),str(request_path)],
                                     capture_output=True,text=True,encoding="utf-8")
            if process.returncode:
                raise RuntimeError(process.stderr or process.stdout)
            print(process.stdout.splitlines()[-1])
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1)


def record_experiment(config, match_existing=False):
    write_request(build_request(config,match_existing))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("code_dir",type=Path)
    parser.add_argument("--match-existing",action="store_true")
    args=parser.parse_args()
    code_dir=args.code_dir.resolve()
    os.chdir(code_dir)
    from types import SimpleNamespace
    config=SimpleNamespace(**runpy.run_path(str(code_dir / "cfgs.py")))
    record_experiment(config,args.match_existing)
