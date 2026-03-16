import argparse
import os
import random
import shutil
from typing import Dict, List

from tqdm import tqdm

from .data import list_pairs
from .eval import evaluate
from .train_boundary import train as train_boundary
from .train_classifier import train as train_classifier


def run_trial(data_dir: str, out_dir: str, arch: str, loss: str, class_weight: bool, epochs: int) -> Dict:
    boundary_path = os.path.join(out_dir, f"boundary_{arch}.pt")
    classifier_path = os.path.join(out_dir, f"classifier_{loss}_{int(class_weight)}.pt")
    train_boundary(data_dir, boundary_path, epochs, 8, 1e-3, arch)
    train_classifier(data_dir, classifier_path, epochs, 16, 1e-3, loss, 2.0, 0.0, class_weight, use_functional_map=True)
    metrics = evaluate(data_dir, boundary_path, classifier_path, 0.5, use_functional_map=True)
    metrics["boundary_ckpt"] = boundary_path
    metrics["classifier_ckpt"] = classifier_path
    metrics["arch"] = arch
    metrics["loss"] = loss
    metrics["class_weight"] = class_weight
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_limit", type=int, default=0)
    args = parser.parse_args()

    pairs = list_pairs(args.data_dir)
    if not pairs:
        raise SystemExit("dataset not found or empty")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.limit > 0:
        pairs = pairs[: args.limit]

    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("val_ratio + test_ratio must be < 1")

    random.Random(args.seed).shuffle(pairs)
    train_end = int(len(pairs) * (1 - args.val_ratio - args.test_ratio))
    val_end = int(len(pairs) * (1 - args.test_ratio))
    train_pairs = pairs[:train_end]
    val_pairs = pairs[train_end:val_end]
    test_pairs = pairs[val_end:]

    train_dir = os.path.join(args.out_dir, "train")
    val_dir = os.path.join(args.out_dir, "val")
    test_dir = os.path.join(args.out_dir, "test")
    for root, subset in [(train_dir, train_pairs), (val_dir, val_pairs), (test_dir, test_pairs)]:
        audio_dir = os.path.join(root, "audio")
        ann_dir = os.path.join(root, "annotations")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        for audio_path, ann_path in subset:
            shutil.copy(audio_path, os.path.join(audio_dir, os.path.basename(audio_path)))
            shutil.copy(ann_path, os.path.join(ann_dir, os.path.basename(ann_path)))
    trials = []
    arches = ["transformer", "lstm"]
    losses = ["focal", "ce"]
    weights = [True, False]
    if args.quick:
        arches = ["transformer"]
        losses = ["focal"]
        weights = [True]

    for arch in arches:
        for loss in losses:
            for class_weight in weights:
                trials.append(run_trial(train_dir, args.out_dir, arch, loss, class_weight, args.epochs))

    evals = []
    for trial in tqdm(trials, desc="validate"):
        metrics = evaluate(val_dir, trial["boundary_ckpt"], trial["classifier_ckpt"], 0.5, args.eval_limit, use_functional_map=True)
        metrics.update(trial)
        evals.append(metrics)

    best = max(evals, key=lambda x: x["boundary_f1"] + x["label_acc"])
    shutil.copy(best["boundary_ckpt"], os.path.join(args.out_dir, "best_boundary.pt"))
    shutil.copy(best["classifier_ckpt"], os.path.join(args.out_dir, "best_classifier.pt"))
    test_metrics = evaluate(test_dir, best["boundary_ckpt"], best["classifier_ckpt"], 0.5, args.eval_limit, use_functional_map=True)
    best["test_boundary_f1"] = test_metrics["boundary_f1"]
    best["test_label_acc"] = test_metrics["label_acc"]
    print(best)


if __name__ == "__main__":
    main()
