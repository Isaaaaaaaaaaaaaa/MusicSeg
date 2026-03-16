from typing import Dict, List, Tuple

import numpy as np


def f1_score(pred: List[int], target: List[int]) -> float:
    pred = np.array(pred)
    target = np.array(target)
    tp = np.sum((pred == 1) & (target == 1))
    fp = np.sum((pred == 1) & (target == 0))
    fn = np.sum((pred == 0) & (target == 1))
    if tp + fp + fn == 0:
        return 0.0
    return (2 * tp) / (2 * tp + fp + fn)


def boundary_retrieval_fmeasure(
    pred_boundaries: List[float],
    gt_boundaries: List[float],
    tolerance: float = 0.5,
) -> Dict[str, float]:
    """
    边界检索 F-measure (Boundary Retrieval F-measure)

    计算预测的边界点与真实边界点的匹配程度。
    如果预测边界与真实边界的距离在 tolerance 范围内，则认为匹配成功。

    Args:
        pred_boundaries: 预测的边界时间点列表（秒）
        gt_boundaries: 真实的边界时间点列表（秒）
        tolerance: 匹配容差（秒），默认0.5秒

    Returns:
        包含 precision, recall, f_measure 的字典
    """
    if not gt_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f_measure": 0.0}

    if not pred_boundaries:
        return {"precision": 0.0, "recall": 0.0, "f_measure": 0.0}

    matched_gt = set()
    tp = 0

    for pred in pred_boundaries:
        for i, gt in enumerate(gt_boundaries):
            if i in matched_gt:
                continue
            if abs(pred - gt) <= tolerance:
                matched_gt.add(i)
                tp += 1
                break

    fp = len(pred_boundaries) - tp
    fn = len(gt_boundaries) - tp

    precision = tp / len(pred_boundaries) if pred_boundaries else 0.0
    recall = tp / len(gt_boundaries) if gt_boundaries else 0.0
    f_measure = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_measure": f_measure,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def framewise_accuracy(pred: np.ndarray, target: np.ndarray, ignore_index: int = -100) -> float:
    pred = np.asarray(pred)
    target = np.asarray(target)
    valid = target != int(ignore_index)
    if valid.sum() == 0:
        return 0.0
    return float((pred[valid] == target[valid]).mean())


def pairwise_f_score(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
) -> Dict[str, float]:
    """
    基于对的 F-score (Pairwise F-score)

    评估段落标注的准确性。对于每一对时间点，检查它们是否属于
    相同的段落（在预测和真实标注中）。

    Args:
        pred_segments: 预测的段落列表，每个段落包含 start, end, label
        gt_segments: 真实的段落列表，每个段落包含 start, end, label

    Returns:
        包含 precision, recall, f_score 的字典
    """
    if not pred_segments or not gt_segments:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    # 获取所有段落的边界时间点
    pred_times = set()
    for seg in pred_segments:
        pred_times.add(seg["start"])
        pred_times.add(seg["end"])
    pred_times = sorted(pred_times)

    gt_times = set()
    for seg in gt_segments:
        gt_times.add(seg["start"])
        gt_times.add(seg["end"])
    gt_times = sorted(gt_times)

    # 合并所有时间点
    all_times = sorted(set(pred_times) | set(gt_times))

    def get_label_at_time(segments: List[Dict], time: float) -> str:
        """获取指定时间点所属的段落标签"""
        for seg in segments:
            if seg["start"] <= time < seg["end"]:
                return seg["label"]
        return ""

    # 构建标签序列
    pred_labels = [get_label_at_time(pred_segments, t) for t in all_times]
    gt_labels = [get_label_at_time(gt_segments, t) for t in all_times]

    # 计算成对一致性
    n = len(all_times)
    tp = 0  # 预测和真实都认为在同一段落
    fp = 0  # 预测认为在同一段落，但真实不同
    fn = 0  # 真实在同一段落，但预测不同

    for i in range(n):
        for j in range(i + 1, n):
            pred_same = pred_labels[i] == pred_labels[j]
            gt_same = gt_labels[i] == gt_labels[j]

            if pred_same and gt_same:
                tp += 1
            elif pred_same and not gt_same:
                fp += 1
            elif not pred_same and gt_same:
                fn += 1

    total_pairs = tp + fp + fn
    if total_pairs == 0:
        return {"precision": 0.0, "recall": 0.0, "f_score": 0.0}

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f_score = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f_score": f_score,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def normalized_conditional_entropy(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
) -> Dict[str, float]:
    """
    归一化条件熵 (Normalized Conditional Entropy)

    衡量预测标注相对于真实标注的不确定性。
    包括两个方向：
    - Over-segmentation: 给定真实段落时预测段落的不确定性
    - Under-segmentation: 给定预测段落时真实段落的不确定性

    Args:
        pred_segments: 预测的段落列表
        gt_segments: 真实的段落列表

    Returns:
        包含 over_segmentation, under_segmentation, total_error 的字典
    """
    if not pred_segments or not gt_segments:
        return {
            "over_segmentation": 1.0,
            "under_segmentation": 1.0,
            "total_error": 1.0,
        }

    # 获取所有时间点
    pred_times = set()
    for seg in pred_segments:
        pred_times.add(seg["start"])
        pred_times.add(seg["end"])

    gt_times = set()
    for seg in gt_segments:
        gt_times.add(seg["start"])
        gt_times.add(seg["end"])

    all_times = sorted(set(pred_times) | set(gt_times))

    def get_label_at_time(segments: List[Dict], time: float) -> str:
        for seg in segments:
            if seg["start"] <= time < seg["end"]:
                return seg["label"]
        return ""

    # 构建标签序列
    pred_labels = [get_label_at_time(pred_segments, t) for t in all_times]
    gt_labels = [get_label_at_time(gt_segments, t) for t in all_times]

    # 计算联合分布
    joint_counts: Dict[Tuple[str, str], int] = {}
    pred_counts: Dict[str, int] = {}
    gt_counts: Dict[str, int] = {}
    total = len(all_times)

    for p, g in zip(pred_labels, gt_labels):
        joint_counts[(p, g)] = joint_counts.get((p, g), 0) + 1
        pred_counts[p] = pred_counts.get(p, 0) + 1
        gt_counts[g] = gt_counts.get(g, 0) + 1

    if total == 0:
        return {
            "over_segmentation": 1.0,
            "under_segmentation": 1.0,
            "total_error": 1.0,
        }

    # 计算条件熵 H(P|G) - Over-segmentation
    h_pg = 0.0
    for g, count_g in gt_counts.items():
        if count_g == 0:
            continue
        p_g = count_g / total
        for p in pred_counts:
            count_pg = joint_counts.get((p, g), 0)
            if count_pg > 0:
                p_pg = count_pg / count_g
                h_pg -= p_g * p_pg * np.log2(p_pg)

    # 计算条件熵 H(G|P) - Under-segmentation
    h_gp = 0.0
    for p, count_p in pred_counts.items():
        if count_p == 0:
            continue
        p_p = count_p / total
        for g in gt_counts:
            count_pg = joint_counts.get((p, g), 0)
            if count_pg > 0:
                p_gp = count_pg / count_p
                h_gp -= p_p * p_gp * np.log2(p_gp)

    # 归一化（使用真实标注的熵作为分母）
    h_g = 0.0
    for g, count_g in gt_counts.items():
        if count_g > 0:
            p_g = count_g / total
            h_g -= p_g * np.log2(p_g)

    h_p = 0.0
    for p, count_p in pred_counts.items():
        if count_p > 0:
            p_p = count_p / total
            h_p -= p_p * np.log2(p_p)

    # 避免除以0
    over_seg = h_pg / h_g if h_g > 0 else 0.0
    under_seg = h_gp / h_p if h_p > 0 else 0.0
    total_error = over_seg + under_seg

    return {
        "over_segmentation": min(1.0, over_seg),
        "under_segmentation": min(1.0, under_seg),
        "total_error": min(2.0, total_error),
    }


def evaluate_all_metrics(
    pred_segments: List[Dict],
    gt_segments: List[Dict],
    tolerance: float = 0.5,
) -> Dict[str, float]:
    """
    计算所有评估指标

    Args:
        pred_segments: 预测的段落列表
        gt_segments: 真实的段落列表
        tolerance: 边界匹配的容差（秒）

    Returns:
        包含所有指标的字典
    """
    # 提取边界时间点
    pred_bounds = [seg["start"] for seg in pred_segments[1:]]  # 跳过第一个0
    gt_bounds = [seg["start"] for seg in gt_segments[1:]]

    # 边界检索 F-measure
    boundary_metrics = boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance)

    # Pairwise F-score
    pairwise_metrics = pairwise_f_score(pred_segments, gt_segments)

    # 归一化条件熵
    entropy_metrics = normalized_conditional_entropy(pred_segments, gt_segments)

    return {
        # 边界检测指标
        "boundary_precision": boundary_metrics["precision"],
        "boundary_recall": boundary_metrics["recall"],
        "boundary_f_measure": boundary_metrics["f_measure"],
        # 段落标注指标
        "pairwise_precision": pairwise_metrics["precision"],
        "pairwise_recall": pairwise_metrics["recall"],
        "pairwise_f_score": pairwise_metrics["f_score"],
        # 条件熵指标
        "over_segmentation": entropy_metrics["over_segmentation"],
        "under_segmentation": entropy_metrics["under_segmentation"],
        "total_entropy_error": entropy_metrics["total_error"],
    }
