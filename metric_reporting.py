"""Console reporting from existing metric CSVs; no training or result writes."""
import csv


def print_metric_row(stage, row):
    print(f"[{stage}] Checkpoint: {row['Checkpoint']}")
    parts = []
    for label in ["Dice", "IoU", "HD95", "Precision", "Recall"]:
        key = label if label in row else f"{label.lower()}_macro_avg"
        if key not in row:
            continue
        value = float(row[key])
        formatted = f"{value:.4f}" if label == "HD95" else f"{value * 100:.2f}%"
        parts.append(f"{label}={formatted}")
    print(f"[{stage}] " + " | ".join(parts))


def print_saved_metrics(stage, csv_path, checkpoint):
    with open(csv_path, newline="", encoding="utf-8") as stream:
        matches = [row for row in csv.DictReader(stream) if row["Checkpoint"] == checkpoint]
    if not matches:
        raise RuntimeError(f"No saved metrics for {checkpoint} in {csv_path}")
    print_metric_row(stage, matches[-1])
