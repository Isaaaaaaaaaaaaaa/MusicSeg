import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from .config import AudioConfig, BoundaryConfig, TransformerConfig
from .data import BoundaryDataset, ClassifierDataset, list_pairs, list_pairs_by_split, load_segments, infer_functional_segments, boundary_labels
from .model import BoundaryNet, MultiScaleTransformerBoundaryNet, MultiScaleTransformerBoundaryNetV2, MultiScaleTransformerBoundaryNetV3, TransformerBoundaryNet, MultiResolutionBoundaryNet, SegmentClassifier, save_boundary, save_classifier, TVLoss1D
from .metrics import boundary_retrieval_fmeasure, pairwise_f_score, normalized_conditional_entropy
from .infer import analyze, refine_peaks
from scipy.ndimage import gaussian_filter1d


def split_pairs(pairs: List[Tuple[str, str]], val_ratio: float, test_ratio: float, seed: int):
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test : n_test + n_val]
    train_pairs = pairs[n_test + n_val :]
    return train_pairs, val_pairs, test_pairs


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    targets = targets.to(dtype=logits.dtype)
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    alpha_t = float(alpha) * targets + (1.0 - float(alpha)) * (1.0 - targets)
    loss = alpha_t * (1.0 - p_t).pow(float(gamma)) * ce
    return loss.mean()


class StructureContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 0.5, max_dist: int = 16):
        super().__init__()
        self.margin = margin
        self.max_dist = max_dist

    def forward(self, features: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
        # features: (B, T, D)
        # boundaries: (B, T) - 1.0 at boundary, 0.0 otherwise
        
        B, T, D = features.shape
        if T < 2:
            return torch.tensor(0.0, device=features.device)
            
        # Sample pairs (simplified for efficiency)
        loss = torch.tensor(0.0, device=features.device)
        count = 0
        
        distances = torch.randint(1, self.max_dist + 1, (4,), device=features.device).unique()
        
        for d in distances:
            d = int(d)
            if T <= d:
                continue
                
            f1 = features[:, :-d, :]
            f2 = features[:, d:, :]
            
            b_binary = (boundaries > 0.5).float()
            b_cum = torch.cumsum(b_binary, dim=1)
            b_diff = b_cum[:, d:] - b_cum[:, :-d]
            is_boundary = (b_diff > 0.5) | (b_binary[:, :-d] > 0.5) | (b_binary[:, d:] > 0.5)
            
            sim = F.cosine_similarity(f1, f2, dim=-1)
            
            pos_mask = (~is_boundary).float()
            pos_loss = (1.0 - sim) * pos_mask
            
            neg_mask = is_boundary.float()
            neg_loss = F.relu(sim - self.margin) * neg_mask
            
            loss += (pos_loss.sum() + neg_loss.sum()) / (B * (T - d))
            count += 1
            
        return loss / max(1, count)

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, T) or (B, T, 1)
        # targets: (B, T) or (B, T, 1)
        
        probs = torch.sigmoid(logits)
        
        # Flatten
        probs = probs.view(-1)
        targets = targets.view(-1)
        
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1. - dice


def evaluate_boundary_with_metrics(
    model: torch.nn.Module,
    pairs: List[Tuple[str, str]],
    audio_cfg: AudioConfig,
    boundary_cfg: BoundaryConfig,
    device: torch.device,
    tolerance: float = 0.5,
    thresholds: List[float] = None,
) -> Dict:
    """
    使用标准音乐结构分析指标评估边界检测模型
    """
    model.eval()
    thresholds = thresholds or [boundary_cfg.threshold]
    metrics_by_thr = {
        thr: {
            "boundary": [],
            "pairwise": [],
            "entropy": [],
        }
        for thr in thresholds
    }

    with torch.no_grad():
        for audio_path, ann_path in tqdm(pairs, desc="evaluating", leave=False):
            # 加载真实标注
            segments = load_segments(ann_path)
            gt_segments = [s.__dict__ for s in segments]
            gt_bounds = [seg["start"] for seg in gt_segments[1:]]

            # 加载音频并预测
            from .infer import compute_mel, sliding_boundary_probs, hierarchical_boundary_probs, peak_pick
            import librosa

            y = None
            mel = None
            duration = float(gt_segments[-1]["end"]) if gt_segments else 0.0
            if str(audio_path).lower().endswith(".npz"):
                data = np.load(audio_path, allow_pickle=True)
                mel = data["data"]
                mel = librosa.power_to_db(mel)
                mel = (mel - mel.mean()) / (mel.std() + 1e-6)
            else:
                y, _ = librosa.load(audio_path, sr=audio_cfg.sample_rate, mono=True)
                mel = compute_mel(y, audio_cfg)
                duration = len(y) / audio_cfg.sample_rate

            if boundary_cfg.use_hierarchical:
                probs = hierarchical_boundary_probs(model, mel, boundary_cfg)
            else:
                use_global = hasattr(model, "local_frames")
                probs = sliding_boundary_probs(model, mel, boundary_cfg, use_global=use_global)

            beat_times = None
            if y is not None and getattr(boundary_cfg, "beat_snap", False):
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
                beat_times = librosa.frames_to_time(beat_frames, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
            for thr in thresholds:
                boundary_cfg.threshold = float(thr)
                if getattr(boundary_cfg, "use_songformer_postprocess", False):
                    picks = songformer_pick_boundaries(probs, audio_cfg, boundary_cfg)
                else:
                    picks = peak_pick(
                        probs,
                        thr,
                        boundary_cfg.min_distance,
                        getattr(boundary_cfg, "smooth_window", 1),
                        getattr(boundary_cfg, "prominence", 0.0),
                        getattr(boundary_cfg, "max_peaks", 0),
                    )
                pred_bounds = [p * audio_cfg.hop_length / audio_cfg.sample_rate for p in picks]
                edge_margin_sec = float(getattr(boundary_cfg, "edge_margin_sec", 0.0) or 0.0)
                if edge_margin_sec > 0:
                    pred_bounds = [t for t in pred_bounds if edge_margin_sec <= t <= duration - edge_margin_sec]
                if beat_times is not None and len(pred_bounds) > 0 and len(beat_times) > 0:
                    max_diff = float(getattr(boundary_cfg, "beat_snap_max", 0.35))
                    snapped = []
                    for t in pred_bounds:
                        j = int(np.argmin(np.abs(beat_times - t)))
                        bt = float(beat_times[j])
                        snapped.append(bt if abs(bt - t) <= max_diff else float(t))
                    pred_bounds = sorted(set(snapped))

                pred_segments = [{"start": 0.0, "end": duration, "label": "unknown"}]
                if pred_bounds:
                    pred_segments = []
                    times = [0.0] + pred_bounds + [duration]
                    for i in range(len(times) - 1):
                        pred_segments.append({
                            "start": times[i],
                            "end": times[i + 1],
                            "label": f"seg_{i}"
                        })

                boundary_metrics = boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance)
                pairwise_metrics = pairwise_f_score(pred_segments, gt_segments)
                entropy_metrics = normalized_conditional_entropy(pred_segments, gt_segments)
                metrics_by_thr[thr]["boundary"].append(boundary_metrics)
                metrics_by_thr[thr]["pairwise"].append(pairwise_metrics)
                metrics_by_thr[thr]["entropy"].append(entropy_metrics)

    # 计算平均指标
    def avg_dicts(dicts):
        if not dicts:
            return {}
        keys = dicts[0].keys()
        return {k: float(np.mean([d.get(k, 0) for d in dicts])) for k in keys}

    summary = {}
    for thr, items in metrics_by_thr.items():
        summary[thr] = {
            "boundary": avg_dicts(items["boundary"]),
            "pairwise": avg_dicts(items["pairwise"]),
            "entropy": avg_dicts(items["entropy"]),
        }

    def score(t: float) -> float:
        b = summary[t]["boundary"].get("f_measure", 0.0)
        p = summary[t]["pairwise"].get("f_score", 0.0)
        over = summary[t]["entropy"].get("over_segmentation", 0.0)
        total_err = summary[t]["entropy"].get("total_error", 0.0)
        return float((1.0 * b) + (2.0 * p) - (0.5 * over) - (0.2 * total_err))

    best_thr = max(summary.keys(), key=score)
    return {
        "boundary": summary[best_thr]["boundary"],
        "pairwise": summary[best_thr]["pairwise"],
        "entropy": summary[best_thr]["entropy"],
        "best_threshold": best_thr,
        "best_score": score(best_thr),
    }


def evaluate_boundary_with_paper_metrics(
    model: torch.nn.Module,
    pairs: List[Tuple[str, str]],
    audio_cfg: AudioConfig,
    boundary_cfg: BoundaryConfig,
    device: torch.device,
    thresholds: List[float] = None,
) -> Dict:
    model.eval()
    thresholds = thresholds or [boundary_cfg.threshold]
    thresholds = [float(t) for t in thresholds]
    hr5_by_thr: Dict[float, List[float]] = {t: [] for t in thresholds}
    hr3_by_thr: Dict[float, List[float]] = {t: [] for t in thresholds}
    pred_counts: Dict[float, List[int]] = {t: [] for t in thresholds}
    gt_counts: Dict[float, List[int]] = {t: [] for t in thresholds}

    from .infer import compute_mel, sliding_boundary_probs, hierarchical_boundary_probs, peak_pick, limit_peaks, refine_peaks, songformer_pick_boundaries
    import librosa

    hop_sec = float(audio_cfg.hop_length / float(audio_cfg.sample_rate))

    with torch.no_grad():
        for audio_path, ann_path in tqdm(pairs, desc="paper-evaluating", leave=False):
            segments = load_segments(ann_path)
            gt_bounds = [float(s.start) for s in segments[1:]]
            duration = float(segments[-1].end) if segments else 0.0

            y = None
            if str(audio_path).lower().endswith(".npz"):
                data = np.load(audio_path, allow_pickle=True)
                mel = data["data"]
                mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10)
                mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
                mel = (mel - float(mel.mean())) / (float(mel.std()) + 1e-6)
                mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                y, _ = librosa.load(audio_path, sr=audio_cfg.sample_rate, mono=True)
                mel = compute_mel(y, audio_cfg)

            if boundary_cfg.use_hierarchical:
                probs = hierarchical_boundary_probs(model, mel, boundary_cfg)
            else:
                probs = sliding_boundary_probs(model, mel, boundary_cfg)

            beat_times = None
            if y is not None and getattr(boundary_cfg, "beat_snap", False):
                tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
                beat_times = librosa.frames_to_time(beat_frames, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
            for thr in thresholds:
                boundary_cfg.threshold = float(thr)
                if getattr(boundary_cfg, "use_songformer_postprocess", False):
                    picks = songformer_pick_boundaries(probs, audio_cfg, boundary_cfg)
                else:
                    picks = peak_pick(
                        probs,
                        float(thr),
                        int(boundary_cfg.min_distance),
                        int(getattr(boundary_cfg, "smooth_window", 1)),
                        float(getattr(boundary_cfg, "prominence", 0.0)),
                        int(getattr(boundary_cfg, "max_peaks", 0) or 0),
                    )
                    picks = refine_peaks(probs, picks, radius=int(getattr(boundary_cfg, "refine_radius", 6)))
                edge_margin_sec = float(getattr(boundary_cfg, "edge_margin_sec", 0.0) or 0.0)
                if edge_margin_sec > 0 and duration > 0:
                    em = int(round(edge_margin_sec * audio_cfg.sample_rate / audio_cfg.hop_length))
                    picks = [int(p) for p in picks if int(p) >= em and int(p) <= int(probs.shape[0]) - 1 - em]
                max_peaks = int(getattr(boundary_cfg, "max_peaks", 0) or 0)
                if max_peaks <= 0:
                    max_peaks = max(4, int(duration / 20.0)) if duration > 0 else 0
                if max_peaks > 0:
                    picks = limit_peaks(picks, probs, int(boundary_cfg.min_distance), int(max_peaks))
                pred_bounds = [float(p) * hop_sec for p in picks]
                if beat_times is not None and len(pred_bounds) > 0 and len(beat_times) > 0:
                    max_diff = float(getattr(boundary_cfg, "beat_snap_max", 0.35))
                    r = int(getattr(boundary_cfg, "beat_refine_radius_frames", 0) or 0)
                    snapped = []
                    for t in pred_bounds:
                        j = int(np.argmin(np.abs(beat_times - t)))
                        bt = float(beat_times[j])
                        if abs(bt - t) <= max_diff and r > 0:
                            b_idx = int(round(bt / hop_sec))
                            left = max(0, b_idx - r)
                            right = min(len(probs) - 1, b_idx + r)
                            k = int(left + np.argmax(probs[left : right + 1]))
                            snapped.append(float(k) * hop_sec)
                        else:
                            snapped.append(bt if abs(bt - t) <= max_diff else float(t))
                    pred_bounds = sorted(set(snapped))
                pred_counts[thr].append(int(len(pred_bounds)))
                gt_counts[thr].append(int(len(gt_bounds)))
                hr5_by_thr[thr].append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=0.5).get("f_measure", 0.0)))
                hr3_by_thr[thr].append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=3.0).get("f_measure", 0.0)))

    summary: Dict[float, Dict] = {}
    for thr in thresholds:
        summary[thr] = {
            "HR.5F": float(np.mean(hr5_by_thr[thr])) if hr5_by_thr[thr] else 0.0,
            "HR3F": float(np.mean(hr3_by_thr[thr])) if hr3_by_thr[thr] else 0.0,
            "pred_bounds_per_song": float(np.mean(pred_counts[thr])) if pred_counts[thr] else 0.0,
            "gt_bounds_per_song": float(np.mean(gt_counts[thr])) if gt_counts[thr] else 0.0,
        }

    w5 = 0.7
    w3 = 0.3
    best_thr = max(summary.keys(), key=lambda t: (w5 * summary[t]["HR.5F"] + w3 * summary[t]["HR3F"]))
    return {
        "paper": summary[best_thr],
        "best_threshold": float(best_thr),
        "best_score": float(w5 * summary[best_thr]["HR.5F"] + w3 * summary[best_thr]["HR3F"]),
        "summary": summary,
    }


def estimate_pos_weight(pairs: List[Tuple[str, str]], audio_cfg: AudioConfig, boundary_cfg: BoundaryConfig, max_value: float = 100.0) -> float:
    total_frames = 0
    total_pos = 0.0
    for audio_path, ann_path in pairs:
        try:
            segments = load_segments(ann_path)
            if not segments:
                continue
            duration = float(segments[-1].end)
            if duration <= 0:
                continue
            n_frames = int(duration * audio_cfg.sample_rate / audio_cfg.hop_length)
            if n_frames <= 0:
                continue
            labels = boundary_labels(
                segments,
                n_frames,
                audio_cfg,
                getattr(boundary_cfg, "label_radius", 0),
                getattr(boundary_cfg, "label_sigma", 0.0),
            )
            total_frames += len(labels)
            total_pos += float(labels.sum())
        except Exception:
            continue
    if total_frames <= 0 or total_pos <= 0:
        return 50.0
    total_neg = max(total_frames - total_pos, 1.0)
    ratio = total_neg / total_pos
    return float(min(float(max_value), max(1.0, ratio)))


def estimate_pos_weight_from_dataset(dataset: torch.utils.data.Dataset, max_value: float = 100.0, samples: int = 256, seed: int = 42) -> float:
    n = int(len(dataset))
    if n <= 0:
        return 50.0
    samples = int(max(1, min(int(samples), n)))
    rng = np.random.RandomState(int(seed))
    idxs = rng.choice(n, size=samples, replace=False)
    total_frames = 0
    total_pos = 0.0
    for i in idxs:
        try:
            item = dataset[int(i)]
            lab = item["labels"]
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            lab = np.asarray(lab, dtype=np.float32)
            total_frames += int(lab.size)
            total_pos += float(lab.sum())
        except Exception:
            continue
    if total_frames <= 0 or total_pos <= 0:
        return 50.0
    total_neg = max(total_frames - total_pos, 1.0)
    ratio = total_neg / total_pos
    return float(min(float(max_value), max(1.0, ratio)))


def estimate_pos_rate_from_dataset(dataset: torch.utils.data.Dataset, samples: int = 256, seed: int = 42) -> float:
    n = int(len(dataset))
    if n <= 0:
        return 0.01
    samples = int(max(1, min(int(samples), n)))
    rng = np.random.RandomState(int(seed))
    idxs = rng.choice(n, size=samples, replace=False)
    total_frames = 0
    total_pos = 0.0
    for i in idxs:
        try:
            item = dataset[int(i)]
            lab = item["labels"]
            if isinstance(lab, torch.Tensor):
                lab = lab.detach().cpu().numpy()
            lab = np.asarray(lab, dtype=np.float32)
            total_frames += int(lab.size)
            total_pos += float(lab.sum())
        except Exception:
            continue
    if total_frames <= 0:
        return 0.01
    rate = float(total_pos) / float(total_frames)
    return float(min(0.5, max(1e-4, rate)))


def set_output_bias(model: nn.Module, prior: float) -> None:
    p = float(min(1.0 - 1e-4, max(1e-4, prior)))
    logit = float(np.log(p / (1.0 - p)))
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear) and model.fc.bias is not None:
        with torch.no_grad():
            model.fc.bias.fill_(logit)


def train_boundary_with_metrics(
    data_dir: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    lr: float,
    arch: str = "transformer",
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    eval_tolerance: float = 0.5,
    eval_thresholds: List[float] = None,
    eval_interval: int = 1,
    eval_mode: str = "standard",
    tv_weight: float = 0.0,
    sparsity_weight: float = 0.0,
    rate_weight: float = 0.0,
    weight_decay: float = 0.01,
    beta1: float = 0.9,
    beta2: float = 0.999,
    grad_clip: float = 1.0,
    warmup_steps: int = 300,
    ema_decay: float = 0.999,
    ema_update_after_steps: int = 0,
    ema_eval: bool = False,
    reg_ramp_epochs: int = 4,
    peak_refine_radius: int = 6,
    use_songformer_postprocess: bool = False,
    local_maxima_filter_size: int = 3,
    postprocess_window_past_sec: float = 12.0,
    postprocess_window_future_sec: float = 12.0,
    postprocess_downsample_factor: int = 3,
    pos_weight_max: float = 100.0,
    boundary_loss: str = "bce",
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    focal_warmup_epochs: int = 3,
    focal_ramp_epochs: int = 6,
    early_stopping_patience: int = 3,
    accum_steps: int = 1,
    contrastive_weight: float = 0.1,
    audio_sample_rate: int = None,
    audio_n_mels: int = None,
    audio_hop_length: int = None,
    audio_n_fft: int = None,
):
    """
    训练边界检测模型，使用标准音乐结构分析指标进行评估
    """
    audio_cfg = AudioConfig(
        sample_rate=int(audio_sample_rate) if audio_sample_rate is not None else AudioConfig.sample_rate,
        n_mels=int(audio_n_mels) if audio_n_mels is not None else AudioConfig.n_mels,
        hop_length=int(audio_hop_length) if audio_hop_length is not None else AudioConfig.hop_length,
        n_fft=int(audio_n_fft) if audio_n_fft is not None else AudioConfig.n_fft,
    )
    boundary_cfg = BoundaryConfig()
    boundary_cfg.refine_radius = int(peak_refine_radius)
    boundary_cfg.use_songformer_postprocess = bool(use_songformer_postprocess)
    boundary_cfg.local_maxima_filter_size = int(local_maxima_filter_size)
    boundary_cfg.postprocess_window_past_sec = float(postprocess_window_past_sec)
    boundary_cfg.postprocess_window_future_sec = float(postprocess_window_future_sec)
    boundary_cfg.postprocess_downsample_factor = int(postprocess_downsample_factor)
    split = list_pairs_by_split(data_dir)
    if split is not None:
        train_pairs, val_pairs, test_pairs = split
    else:
        pairs = list_pairs(data_dir)
        train_pairs, val_pairs, test_pairs = split_pairs(pairs, val_ratio, test_ratio, seed)

    if arch == "multi_res":
        from .data import MultiResolutionBoundaryDataset
        train_dataset = MultiResolutionBoundaryDataset(
            data_dir, audio_cfg, boundary_cfg, pairs=train_pairs,
            local_window_sec=30.0, global_window_sec=420.0
        )
    else:
        train_dataset = BoundaryDataset(data_dir, audio_cfg, boundary_cfg, pairs=train_pairs, augment=True)
    workers = max(1, min(os.cpu_count() or 1, 4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    metrics_path = out_path + ".metrics.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("")
    if arch == "multi_res":
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=False,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=workers,
            pin_memory=pin,
            persistent_workers=workers > 0,
        )

    if arch == "lstm":
        model = BoundaryNet(audio_cfg.n_mels)
        arch_cfg = {"type": "lstm"}
    elif arch == "ms_transformer":
        tcfg = TransformerConfig()
        model = MultiScaleTransformerBoundaryNet(
            n_mels=audio_cfg.n_mels,
            d_model=128,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            long_pool=4,
        )
        arch_cfg = {"type": "ms_transformer", "tcfg": tcfg.__dict__, "long_pool": 4}
    elif arch == "ms_transformer_v2":
        tcfg = TransformerConfig(num_layers=4, nhead=4, dim_feedforward=384, dropout=0.1)
        model = MultiScaleTransformerBoundaryNetV2(
            n_mels=audio_cfg.n_mels,
            d_model=192,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            long_pool=4,
        )
        arch_cfg = {"type": "ms_transformer_v2", "tcfg": tcfg.__dict__, "long_pool": 4, "d_model": 192}
    elif arch == "songformer_ds":
        tcfg = TransformerConfig(num_layers=4, nhead=4, dim_feedforward=384, dropout=0.1)
        model = MultiScaleTransformerBoundaryNetV3(
            n_mels=audio_cfg.n_mels,
            d_model=192,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            downsample_kernel=int(getattr(boundary_cfg, "postprocess_downsample_factor", 3)),
            downsample_stride=int(getattr(boundary_cfg, "postprocess_downsample_factor", 3)),
            downsample_dropout=0.1,
            drop_path_rate=0.1, # SOTA: Stochastic Depth
        )
        arch_cfg = {
            "type": "songformer_ds",
            "tcfg": tcfg.__dict__,
            "d_model": 192,
            "downsample_kernel": int(getattr(boundary_cfg, "postprocess_downsample_factor", 3)),
            "downsample_stride": int(getattr(boundary_cfg, "postprocess_downsample_factor", 3)),
            "downsample_dropout": 0.1,
            "drop_path_rate": 0.1,
        }
    elif arch == "multi_res":
        tcfg = TransformerConfig(num_layers=4, nhead=4, dim_feedforward=512, dropout=0.1)
        model = MultiResolutionBoundaryNet(
            n_mels=audio_cfg.n_mels,
            d_model=256,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            hop_length=audio_cfg.hop_length,
            sample_rate=audio_cfg.sample_rate,
        )
        arch_cfg = {"type": "multi_res", "tcfg": tcfg.__dict__, "d_model": 256}
    else:
        tcfg = TransformerConfig()
        model = TransformerBoundaryNet(
            n_mels=audio_cfg.n_mels,
            d_model=128,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
        )
        arch_cfg = {"type": "transformer", "tcfg": tcfg.__dict__}

    model.to(device)
    # SongFormer Optimizer Settings
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(weight_decay),
        betas=(float(beta1), float(beta2)),
    )
    total_steps = int(max(1, epochs * max(1, len(train_loader))))
    warmup_steps = int(max(0, warmup_steps))
    if warmup_steps > total_steps:
        warmup_steps = total_steps // 10
    def lr_lambda(step: int) -> float:
        step = int(step)
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    
    # Loss functions
    pos_weight_value = estimate_pos_weight(train_pairs, audio_cfg, boundary_cfg, max_value=float(pos_weight_max))
    if float(pos_weight_value) <= 1.01:
        pos_weight_value = estimate_pos_weight_from_dataset(train_dataset, max_value=float(pos_weight_max), samples=256, seed=seed)
        print("warning: pos_weight from annotations is near 1.0; switched to dataset-based estimate")
    print(f"pos_weight={pos_weight_value:.2f}")
    
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_value, device=device))
    dice_loss_fn = DiceLoss() # SOTA: Dice Loss for segmentation
    contrastive_loss_fn = StructureContrastiveLoss()
    tv_loss_fn = TVLoss1D(beta=0.6, lambda_tv=1.0, boundary_threshold=0.01, reduction_weight=0.1)
    
    prior = estimate_pos_rate_from_dataset(train_dataset, samples=256, seed=seed)
    set_output_bias(model, prior)
    print(f"pos_rate={prior:.6f}")

    best_score = float("-inf")
    best_boundary_f = float("-inf")
    best_hr5 = float("-inf")
    no_improve_count = 0
    use_ema = float(ema_decay) > 0.0 and float(ema_decay) < 1.0
    ema_state = None
    ema_ready = False
    if use_ema:
        ema_state = {k: v.detach().clone().to(device="cpu") for k, v in model.state_dict().items()}

    accum_steps = int(max(1, accum_steps))
    global_step = 0
    
    # Gaussian Smoothing Kernel
    def gaussian_smooth_labels(labels_tensor, num_neighbors=10):
        sigma = num_neighbors / 3.0
        labels_np = labels_tensor.cpu().numpy()
        smoothed = np.zeros_like(labels_np)
        for i in range(labels_np.shape[0]):
            smoothed[i] = gaussian_filter1d(labels_np[i], sigma=sigma)
        # Theoretical max for normalization
        max_val = 1.0 / (np.sqrt(2 * np.pi) * sigma)
        smoothed /= max_val
        smoothed = np.clip(smoothed, 0, 1)
        return torch.from_numpy(smoothed).to(labels_tensor.device).float()

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_loss = 0.0
        total_elems = 0.0
        
        # Ramp up regularization weights
        ramp = 1.0
        if int(reg_ramp_epochs) > 0:
            ramp = min(1.0, float(epoch + 1) / float(int(reg_ramp_epochs)))
            
        opt.zero_grad()
        step_in_epoch = 0
        for batch in tqdm(train_loader, desc=f"boundary epoch {epoch+1}/{epochs}"):
            mel = batch["mel"].to(device)
            labels = batch["labels"].to(device)
            
            # Apply Gaussian Smoothing to labels (SOTA trick)
            # v26_optimized: Disabled by default as it might be too aggressive for HR.5F
            # if epoch < epochs // 2: 
            #     labels = gaussian_smooth_labels(labels)
            
            # Forward pass
            outputs = model(mel)
            
            logits_aux = []
            if isinstance(outputs, dict):
                logits = outputs["out"]
                contrast_feat = outputs.get("contrast", None)
                # Collect aux logits for Deep Supervision
                for k in ["short", "ds", "ds2", "fuse", "global"]:
                    if k in outputs and outputs[k] is not None:
                        logits_aux.append(outputs[k])
            elif isinstance(outputs, tuple):
                logits, contrast_feat = outputs
            else:
                logits = outputs
                contrast_feat = None
            
            # Loss calculation
            loss_mode = str(boundary_loss).lower()
            
            def compute_loss(pred, target):
                # SOTA Improvement: Hybrid Loss with Dynamic Weighting
                if loss_mode == "focal":
                    return sigmoid_focal_loss(
                        pred.float(),
                        target.float(),
                        alpha=float(focal_alpha),
                        gamma=float(focal_gamma),
                    )
                elif loss_mode == "dice":
                    return dice_loss_fn(pred, target)
                elif loss_mode == "bce_dice":
                    bce = loss_fn(pred, target)
                    dice = dice_loss_fn(pred, target)
                    # Dynamic weighting? Or fixed?
                    # SongFormer uses pure BCE or BCE+Dice.
                    # Standard segmentation uses 0.5 * BCE + 0.5 * Dice
                    return 0.5 * bce + 0.5 * dice
                elif loss_mode == "hybrid":
                    bce = loss_fn(pred, target)
                    if int(epoch) < int(focal_warmup_epochs):
                        return bce
                    else:
                        foc = sigmoid_focal_loss(
                            pred.float(),
                            target.float(),
                            alpha=float(focal_alpha),
                            gamma=float(focal_gamma),
                        )
                        # v26_optimized: Smooth transition from BCE to Hybrid
                        # Note: Ensure focal_alpha is high (>0.5) for imbalanced tasks if pos_weight is used in BCE
                        ramp = min(1.0, float(epoch - focal_warmup_epochs) / max(1.0, float(focal_ramp_epochs)))
                        
                        # Fix: If alpha is small (e.g. 0.35) but pos_weight is large (100), they fight.
                        # We force BCE to dominate structure, Focal to refine hard examples.
                        # We weight them 1:1 eventually.
                        return (1.0 - 0.5 * ramp) * bce + (0.5 * ramp) * foc * 10.0 # Scale focal up as it's usually smaller than weighted BCE
                else:
                    return loss_fn(pred, target)

            loss = compute_loss(logits, labels)
            
            # Deep Supervision Loss with decay
            if logits_aux:
                # SOTA: Decay aux loss weight over time
                aux_decay = max(0.1, 1.0 - float(epoch) / float(epochs))
                aux_weight = 0.4 * aux_decay 
                for aux in logits_aux:
                    loss = loss + aux_weight * compute_loss(aux, labels)
            
            # Structure Contrastive Loss (SOTA)
            if contrast_feat is not None and float(contrastive_weight) > 0.0:
                c_loss = contrastive_loss_fn(contrast_feat, labels)
                ramp = min(1.0, float(epoch) / 2.0) 
                loss = loss + (float(contrastive_weight) * ramp) * c_loss
                
            # Regularization losses
            if float(tv_weight) > 0.0:
                # Use SOTA TVLoss1D
                probs = torch.sigmoid(logits.float())
                tv = tv_loss_fn(probs, labels)
                ramp = min(1.0, float(epoch) / 5.0)
                loss = loss + (float(tv_weight) * ramp) * tv
                
            if float(rate_weight) > 0.0:
                probs = torch.sigmoid(logits.float())
                target_rate = labels.float().mean()
                pred_rate = probs.mean()
                ramp = min(1.0, float(epoch) / 5.0)
                loss = loss + (float(rate_weight) * ramp) * (pred_rate - target_rate).pow(2)

            # Gradient accumulation
            loss = loss / float(accum_steps)
            loss.backward()
            step_in_epoch += 1
            
            if step_in_epoch % accum_steps == 0:
                if float(grad_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                opt.step()
                scheduler.step()
                opt.zero_grad()
                global_step += 1
                
                # EMA update
                if use_ema and ema_state is not None and int(global_step) >= int(ema_update_after_steps):
                    d = float(ema_decay)
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            if k in ema_state:
                                ema_state[k].mul_(d).add_(v.detach().to(device="cpu"), alpha=(1.0 - d))
                    ema_ready = True

            total_loss += loss.item() * float(accum_steps) * labels.numel() # Scale back up
            total_elems += labels.numel()

        # Handle last partial batch
        if step_in_epoch % accum_steps != 0:
            if float(grad_clip) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            scheduler.step()
            opt.zero_grad()
            global_step += 1
            if use_ema and ema_state is not None and int(global_step) >= int(ema_update_after_steps):
                 d = float(ema_decay)
                 with torch.no_grad():
                     for k, v in model.state_dict().items():
                         if k in ema_state:
                             ema_state[k].mul_(d).add_(v.detach().to(device="cpu"), alpha=(1.0 - d))
                 ema_ready = True

        train_loss = total_loss / total_elems if total_elems > 0 else 0.0

        do_eval = val_pairs and (
            eval_interval is None
            or int(eval_interval) <= 1
            or ((epoch + 1) % int(eval_interval) == 0)
            or (epoch + 1) == epochs
        )
        # 验证阶段
        if do_eval:
            thresholds = eval_thresholds
            if not thresholds:
                thresholds = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
            eval_model = model
            swapped = False
            # Ensure EMA is used only if ready and requested
            if bool(ema_eval) and use_ema and ema_state is not None and ema_ready:
                cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                # Ensure device consistency
                ema_state_load = {k: v.to(device=device) for k, v in ema_state.items()}
                model.load_state_dict(ema_state_load, strict=True)
                swapped = True
            
            # Set model to eval mode
            model.eval()
            with torch.no_grad():
                if str(eval_mode).lower() == "paper":
                    paper_metrics = evaluate_boundary_with_paper_metrics(
                        eval_model, val_pairs, audio_cfg, boundary_cfg, device, thresholds
                    )
                    paper = paper_metrics.get("paper", {})
                    boundary_f = 0.0
                    pairwise_f = 0.0
                    best_threshold = paper_metrics.get("best_threshold", boundary_cfg.threshold)
                    val_score = float(paper_metrics.get("best_score", 0.0))
                else:
                    val_metrics = evaluate_boundary_with_metrics(
                        eval_model, val_pairs, audio_cfg, boundary_cfg, device, eval_tolerance, thresholds
                    )
                    boundary_f = val_metrics["boundary"].get("f_measure", 0)
                    pairwise_f = val_metrics["pairwise"].get("f_score", 0)
                    best_threshold = val_metrics.get("best_threshold", boundary_cfg.threshold)
                    val_score = float(val_metrics.get("best_score", 0.0))
            
            if swapped:
                model.load_state_dict(cur, strict=True)
            # Restore train mode
            model.train()

            # 记录指标
            if str(eval_mode).lower() == "paper":
                line = {
                    "phase": "val",
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "HR.5F": float(paper.get("HR.5F", 0.0)),
                    "HR3F": float(paper.get("HR3F", 0.0)),
                    "pred_bounds_per_song": float(paper.get("pred_bounds_per_song", 0.0)),
                    "gt_bounds_per_song": float(paper.get("gt_bounds_per_song", 0.0)),
                    "threshold": best_threshold,
                    "score": val_score,
                    "eval_mode": "paper",
                    "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                    "seed": seed,
                }
            else:
                line = {
                    "phase": "val",
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "boundary_precision": val_metrics["boundary"].get("precision", 0),
                    "boundary_recall": val_metrics["boundary"].get("recall", 0),
                    "boundary_f_measure": boundary_f,
                    "pairwise_precision": val_metrics["pairwise"].get("precision", 0),
                    "pairwise_recall": val_metrics["pairwise"].get("recall", 0),
                    "pairwise_f_score": pairwise_f,
                    "over_segmentation": val_metrics["entropy"].get("over_segmentation", 0),
                    "under_segmentation": val_metrics["entropy"].get("under_segmentation", 0),
                    "total_entropy_error": val_metrics["entropy"].get("total_error", 0),
                    "threshold": best_threshold,
                    "score": val_score,
                    "eval_mode": "standard",
                    "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                    "seed": seed,
                }

            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

            if str(eval_mode).lower() == "paper":
                print(
                    f"epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} "
                    f"HR.5F={float(paper.get('HR.5F', 0.0)):.4f} HR3F={float(paper.get('HR3F', 0.0)):.4f} "
                    f"pred/gt={float(paper.get('pred_bounds_per_song', 0.0)):.2f}/{float(paper.get('gt_bounds_per_song', 0.0)):.2f} "
                    f"score={val_score:.4f} thr={best_threshold}"
                )
            else:
                print(
                    f"epoch {epoch+1}/{epochs} train_loss={train_loss:.6f} "
                    f"boundary_f={boundary_f:.4f} pairwise_f={pairwise_f:.4f} "
                    f"over_seg={val_metrics['entropy'].get('over_segmentation', 0):.4f} "
                    f"under_seg={val_metrics['entropy'].get('under_segmentation', 0):.4f} "
                    f"score={val_score:.4f} thr={best_threshold}"
                )

            # 保存最佳模型
            save_reason = None
            
            # SOTA: Score using weighted sum of HR.5F and HR3F (SongFormer metric)
            if str(eval_mode).lower() == "paper":
                hr5_now = float(paper.get("HR.5F", 0.0))
                hr3_now = float(paper.get("HR3F", 0.0))
                combo_now = 0.7 * hr5_now + 0.3 * hr3_now
                
                # Check if this is the best combo score
                if combo_now > best_hr5:
                    best_hr5 = combo_now
                    save_reason = "0.7*HR.5F+0.3*HR3F"
                    
            # Fallback for standard mode
            elif val_score > best_score:
                best_score = val_score
                save_reason = "score"
            elif boundary_f > best_boundary_f:
                best_boundary_f = boundary_f
                if save_reason is None:
                    save_reason = "boundary_f"
                    
            if save_reason is not None:
                no_improve_count = 0
                boundary_cfg.threshold = float(best_threshold)
                
                # Use EMA state for saving if available
                if bool(ema_eval) and use_ema and ema_state is not None and ema_ready:
                    cur = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    # Load EMA weights to model
                    model.load_state_dict({k: v.to(device=device) for k, v in ema_state.items()}, strict=True)
                    save_boundary(
                        out_path,
                        model,
                        {
                            "audio_cfg": audio_cfg.__dict__,
                            "boundary_cfg": boundary_cfg.__dict__,
                            "arch": arch_cfg,
                            "best_metric": best_hr5 if str(eval_mode).lower() == "paper" else best_score,
                        },
                    )
                    # Restore current weights
                    model.load_state_dict(cur, strict=True)
                else:
                    save_boundary(
                        out_path,
                        model,
                        {
                            "audio_cfg": audio_cfg.__dict__,
                            "boundary_cfg": boundary_cfg.__dict__,
                            "arch": arch_cfg,
                            "best_metric": best_hr5 if str(eval_mode).lower() == "paper" else best_score,
                        },
                    )
                if str(eval_mode).lower() == "paper":
                    print(
                        f"  -> Saved best model (HR.5F={float(paper.get('HR.5F', 0.0)):.4f}, "
                        f"HR3F={float(paper.get('HR3F', 0.0)):.4f}, score={val_score:.4f}, thr={best_threshold})"
                    )
                else:
                    print(f"  -> Saved best model ({save_reason}={val_score:.4f}, boundary_f={boundary_f:.4f})")
            else:
                no_improve_count += 1
                if int(early_stopping_patience) > 0 and no_improve_count >= int(early_stopping_patience):
                    print(f"Early stopping triggered after {epoch+1} epochs (no improvement for {no_improve_count} validations)")
                    break

    # 最终测试评估
    if test_pairs:
        thresholds = eval_thresholds
        if not thresholds:
            thresholds = [0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]
        if str(eval_mode).lower() == "paper":
            paper_metrics = evaluate_boundary_with_paper_metrics(
                model, test_pairs, audio_cfg, boundary_cfg, device, thresholds
            )
            paper = paper_metrics.get("paper", {})
            best_threshold = paper_metrics.get("best_threshold", boundary_cfg.threshold)
            line = {
                "phase": "test",
                "epoch": epochs,
                "HR.5F": float(paper.get("HR.5F", 0.0)),
                "HR3F": float(paper.get("HR3F", 0.0)),
                "pred_bounds_per_song": float(paper.get("pred_bounds_per_song", 0.0)),
                "gt_bounds_per_song": float(paper.get("gt_bounds_per_song", 0.0)),
                "threshold": best_threshold,
                "score": float(paper_metrics.get("best_score", 0.0)),
                "eval_mode": "paper",
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }
        else:
            test_metrics = evaluate_boundary_with_metrics(
                model, test_pairs, audio_cfg, boundary_cfg, device, eval_tolerance, thresholds
            )
            best_threshold = test_metrics.get("best_threshold", boundary_cfg.threshold)
            line = {
                "phase": "test",
                "epoch": epochs,
                "boundary_precision": test_metrics["boundary"].get("precision", 0),
                "boundary_recall": test_metrics["boundary"].get("recall", 0),
                "boundary_f_measure": test_metrics["boundary"].get("f_measure", 0),
                "pairwise_precision": test_metrics["pairwise"].get("precision", 0),
                "pairwise_recall": test_metrics["pairwise"].get("recall", 0),
                "pairwise_f_score": test_metrics["pairwise"].get("f_score", 0),
                "over_segmentation": test_metrics["entropy"].get("over_segmentation", 0),
                "under_segmentation": test_metrics["entropy"].get("under_segmentation", 0),
                "total_entropy_error": test_metrics["entropy"].get("total_error", 0),
                "threshold": best_threshold,
                "score": float(test_metrics.get("best_score", 0.0)),
                "eval_mode": "standard",
                "split": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
                "seed": seed,
            }

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print("\n【最终测试结果】")
        if str(eval_mode).lower() == "paper":
            print(f"  HR.5F: {float(line.get('HR.5F', 0.0)):.4f}")
            print(f"  HR3F:  {float(line.get('HR3F', 0.0)):.4f}")
        else:
            print(f"  Boundary F-measure: {test_metrics['boundary'].get('f_measure', 0):.4f}")
            print(f"  Pairwise F-score:   {test_metrics['pairwise'].get('f_score', 0):.4f}")
            print(f"  Total Entropy Error:{test_metrics['entropy'].get('total_error', 0):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="使用标准音乐结构分析指标训练边界检测模型"
    )
    parser.add_argument("--data_dir", required=True, help="数据目录路径")
    parser.add_argument("--out_path", default="checkpoints/boundary.pt", help="模型输出路径")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument(
        "--arch", choices=["lstm", "transformer", "ms_transformer", "ms_transformer_v2"], default="transformer", help="模型架构"
    )
    parser.add_argument("--audio_sample_rate", type=int, default=None, help="采样率覆盖")
    parser.add_argument("--audio_n_mels", type=int, default=None, help="梅尔频带数覆盖")
    parser.add_argument("--audio_hop_length", type=int, default=None, help="hop_length 覆盖")
    parser.add_argument("--audio_n_fft", type=int, default=None, help="n_fft 覆盖")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--eval_tolerance", type=float, default=0.5, help="边界匹配容差（秒）"
    )
    parser.add_argument("--eval_thresholds", default="", help="阈值列表，逗号分隔")
    parser.add_argument("--eval_mode", choices=["standard", "paper"], default="standard")
    parser.add_argument("--eval_interval", type=int, default=1, help="每隔N个epoch评估一次")
    parser.add_argument("--tv_weight", type=float, default=0.0)
    parser.add_argument("--sparsity_weight", type=float, default=0.0)
    parser.add_argument("--rate_weight", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_update_after_steps", type=int, default=0)
    parser.add_argument("--reg_ramp_epochs", type=int, default=4)
    parser.add_argument("--peak_refine_radius", type=int, default=6)
    parser.add_argument("--use_songformer_postprocess", action="store_true", default=False)
    parser.add_argument("--local_maxima_filter_size", type=int, default=3)
    parser.add_argument("--postprocess_window_past_sec", type=float, default=12.0)
    parser.add_argument("--postprocess_window_future_sec", type=float, default=12.0)
    parser.add_argument("--postprocess_downsample_factor", type=int, default=3)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--pos_weight_max", type=float, default=100.0)
    parser.add_argument("--boundary_loss", choices=["bce", "focal", "hybrid"], default="bce")
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_warmup_epochs", type=int, default=3)
    args = parser.parse_args()
    thresholds = None
    if args.eval_thresholds:
        thresholds = []
        for item in str(args.eval_thresholds).split(","):
            if item:
                thresholds.append(float(item))

    train_boundary_with_metrics(
        data_dir=args.data_dir,
        out_path=args.out_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        arch=args.arch,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        eval_tolerance=args.eval_tolerance,
        eval_thresholds=thresholds,
        eval_interval=args.eval_interval,
        eval_mode=args.eval_mode,
        tv_weight=args.tv_weight,
        sparsity_weight=args.sparsity_weight,
        rate_weight=args.rate_weight,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        grad_clip=args.grad_clip,
        warmup_steps=args.warmup_steps,
        ema_decay=args.ema_decay,
        ema_update_after_steps=args.ema_update_after_steps,
        reg_ramp_epochs=args.reg_ramp_epochs,
        peak_refine_radius=args.peak_refine_radius,
        use_songformer_postprocess=args.use_songformer_postprocess,
        local_maxima_filter_size=args.local_maxima_filter_size,
        postprocess_window_past_sec=args.postprocess_window_past_sec,
        postprocess_window_future_sec=args.postprocess_window_future_sec,
        postprocess_downsample_factor=args.postprocess_downsample_factor,
        pos_weight_max=args.pos_weight_max,
        boundary_loss=args.boundary_loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        focal_warmup_epochs=args.focal_warmup_epochs,
        accum_steps=args.accum_steps,
        audio_sample_rate=args.audio_sample_rate,
        audio_n_mels=args.audio_n_mels,
        audio_hop_length=args.audio_hop_length,
        audio_n_fft=args.audio_n_fft,
    )


if __name__ == "__main__":
    main()
