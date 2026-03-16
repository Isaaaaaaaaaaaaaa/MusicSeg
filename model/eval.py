import argparse
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from .data import list_pairs, load_segments, infer_functional_segments
from .infer import analyze
from .metrics import (
    boundary_retrieval_fmeasure,
    pairwise_f_score,
    normalized_conditional_entropy,
    evaluate_all_metrics,
)


def match_boundaries(pred: List[float], gt: List[float], tol: float) -> Tuple[int, int, int]:
    """保持向后兼容的边界匹配函数"""
    matched = set()
    tp = 0
    for p in pred:
        for i, g in enumerate(gt):
            if i in matched:
                continue
            if abs(p - g) <= tol:
                matched.add(i)
                tp += 1
                break
    fp = len(pred) - tp
    fn = len(gt) - tp
    return tp, fp, fn


def f1(tp: int, fp: int, fn: int) -> float:
    """保持向后兼容的F1计算函数"""
    denom = 2 * tp + fp + fn
    if denom == 0:
        return 0.0
    return (2 * tp) / denom


def label_accuracy(pred_segments: List[Dict], gt_segments: List[Dict]) -> float:
    """计算标签准确率（用于向后兼容）"""
    if not gt_segments:
        return 0.0
    correct = 0
    for seg in gt_segments:
        t = (seg["start"] + seg["end"]) / 2
        pred = next((p for p in pred_segments if p["start"] <= t <= p["end"]), None)
        if pred and pred["label"] == seg["label"]:
            correct += 1
    return correct / len(gt_segments)


def evaluate(
    data_dir: str,
    boundary_ckpt: str,
    classifier_ckpt: str,
    tol: float,
    limit: int = 0,
    use_functional_map: bool = True,
) -> Dict:
    """
    使用标准音乐结构分析指标进行评估

    Args:
        data_dir: 数据目录
        boundary_ckpt: 边界检测模型检查点路径
        classifier_ckpt: 分类器模型检查点路径
        tol: 边界匹配容差（秒）
        limit: 限制评估的样本数量（0表示不限制）
        use_functional_map: 是否使用功能性标签映射

    Returns:
        包含所有评估指标的字典
    """
    pairs = list_pairs(data_dir)
    if limit > 0:
        pairs = pairs[:limit]

    # 收集所有指标
    all_metrics = []

    for audio_path, ann_path in tqdm(pairs, desc="eval"):
        segments = load_segments(ann_path)
        if use_functional_map:
            segments = infer_functional_segments(segments)
        gt_segments = [s.__dict__ for s in segments]

        result = analyze(audio_path, boundary_ckpt, classifier_ckpt)
        pred_segments = result["segments"]

        # 计算所有指标
        metrics = evaluate_all_metrics(pred_segments, gt_segments, tol)
        all_metrics.append(metrics)

    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys() if all_metrics else []:
        values = [m[key] for m in all_metrics if key in m]
        avg_metrics[key] = float(np.mean(values)) if values else 0.0

    # 添加样本数量信息
    avg_metrics["num_samples"] = len(pairs)
    avg_metrics["tolerance"] = tol

    return avg_metrics


def print_metrics(metrics: Dict):
    """美观地打印评估指标"""
    print("\n" + "=" * 60)
    print("音乐结构分析评估结果")
    print("=" * 60)

    print("\n【边界检测指标】(Boundary Retrieval)")
    print(f"  Precision:     {metrics.get('boundary_precision', 0):.4f}")
    print(f"  Recall:        {metrics.get('boundary_recall', 0):.4f}")
    print(f"  F-measure:     {metrics.get('boundary_f_measure', 0):.4f}")

    print("\n【段落标注指标】(Pairwise F-score)")
    print(f"  Precision:     {metrics.get('pairwise_precision', 0):.4f}")
    print(f"  Recall:        {metrics.get('pairwise_recall', 0):.4f}")
    print(f"  F-score:       {metrics.get('pairwise_f_score', 0):.4f}")

    print("\n【条件熵指标】(Normalized Conditional Entropy)")
    print(f"  Over-segmentation:   {metrics.get('over_segmentation', 0):.4f}")
    print(f"  Under-segmentation:  {metrics.get('under_segmentation', 0):.4f}")
    print(f"  Total Error:         {metrics.get('total_entropy_error', 0):.4f}")

    print("\n【评估信息】")
    print(f"  样本数量:      {metrics.get('num_samples', 0)}")
    print(f"  容差:          {metrics.get('tolerance', 0):.2f}s")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="使用标准音乐结构分析指标评估模型"
    )
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument(
        "--boundary_ckpt",
        default="checkpoints/boundary.pt",
        help="边界检测模型检查点路径",
    )
    parser.add_argument(
        "--classifier_ckpt",
        default="checkpoints/classifier.pt",
        help="分类器模型检查点路径",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.5,
        help="边界匹配容差（秒），默认0.5秒",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="限制评估的样本数量（0表示不限制）",
    )
    parser.add_argument(
        "--no_functional_map",
        action="store_false",
        dest="use_functional_map",
        help="禁用功能性标签映射",
    )
    parser.set_defaults(use_functional_map=True)
    args = parser.parse_args()

    metrics = evaluate(
        args.data_dir,
        args.boundary_ckpt,
        args.classifier_ckpt,
        args.tolerance,
        args.limit,
        args.use_functional_map,
    )

    print_metrics(metrics)


if __name__ == "__main__":
    main()
