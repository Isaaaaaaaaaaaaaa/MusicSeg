import argparse
import bisect
import json
import math
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch

from .config import AudioConfig, BoundaryConfig
from .data import load_segments, mel_spectrogram, list_pairs_by_split
from .infer import peak_pick, limit_peaks, refine_peaks, load_boundary, load_classifier, sliding_boundary_probs, hierarchical_boundary_probs, songformer_pick_boundaries
from .metrics import boundary_retrieval_fmeasure, framewise_accuracy


IGNORE_INDEX = -100


def map_to_7(label: str) -> str:
    s = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    if s in {"end", "start", "silence"}:
        return "silence"
    if "silence" in s:
        return "silence"
    if "intro" in s:
        return "intro"
    if "outro" in s or "coda" in s or "fade" in s:
        return "outro"
    if "chorus" in s:
        return "chorus"
    if "pre_chorus" in s or "prechorus" in s or "post_chorus" in s:
        return "verse"
    if "bridge" in s:
        return "bridge"
    if "verse" in s:
        return "verse"
    if "solo" in s or "break" in s or "instrument" in s or "interlude" in s or "theme" in s:
        return "inst"
    return "verse"


def labels_7() -> List[str]:
    return ["silence", "intro", "verse", "chorus", "bridge", "outro", "inst"]


def downsample_mode_ceil(x: np.ndarray, factor: int, ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    f = int(factor)
    if f <= 1:
        return x.astype(np.int64)
    t = int(x.shape[0])
    out_t = int(math.ceil(t / float(f)))
    pad = out_t * f - t
    if pad > 0:
        x = np.pad(x, (0, pad), mode="constant", constant_values=int(ignore_index))
    x = x.reshape(out_t, f)
    out = np.full((out_t,), int(ignore_index), dtype=np.int64)
    for i in range(out_t):
        row = x[i]
        row = row[row != int(ignore_index)]
        if row.size == 0:
            continue
        vals, counts = np.unique(row, return_counts=True)
        out[i] = int(vals[int(np.argmax(counts))])
    return out


def frame_hop_sec(audio_cfg: AudioConfig, downsample_factor: int) -> float:
    return float(audio_cfg.hop_length * int(downsample_factor) / float(audio_cfg.sample_rate))


def gt_frame_labels_7(segments, audio_cfg: AudioConfig, duration: float, downsample_factor: int) -> np.ndarray:
    labs = labels_7()
    lab_to_id = {l: i for i, l in enumerate(labs)}
    base_frames = int(math.ceil(float(duration) * audio_cfg.sample_rate / audio_cfg.hop_length))
    base = np.full((base_frames,), int(IGNORE_INDEX), dtype=np.int64)
    for seg in segments:
        lab = map_to_7(seg.label)
        lid = lab_to_id.get(lab, int(IGNORE_INDEX))
        s = int(max(0, math.floor(seg.start * audio_cfg.sample_rate / audio_cfg.hop_length)))
        e = int(max(s + 1, math.ceil(seg.end * audio_cfg.sample_rate / audio_cfg.hop_length)))
        s = min(s, base_frames)
        e = min(e, base_frames)
        if e > s:
            base[s:e] = int(lid)
    ds = downsample_mode_ceil(base, int(downsample_factor), ignore_index=IGNORE_INDEX)
    return ds


def pred_frame_labels_from_segments_7(
    pred_segments: List[Dict], audio_cfg: AudioConfig, duration: float, downsample_factor: int
) -> np.ndarray:
    labs = labels_7()
    lab_to_id = {l: i for i, l in enumerate(labs)}
    hop = frame_hop_sec(audio_cfg, downsample_factor)
    t_out = int(math.ceil(float(duration) / hop))
    y = np.full((t_out,), int(IGNORE_INDEX), dtype=np.int64)
    for seg in pred_segments:
        lab = map_to_7(seg["label"])
        lid = lab_to_id.get(lab, int(IGNORE_INDEX))
        s = float(seg["start"])
        e = float(seg["end"])
        fs = int(max(0, math.floor(s / hop)))
        fe = int(min(t_out, max(fs + 1, math.ceil(e / hop))))
        if fe > fs:
            y[fs:fe] = int(lid)
    return y


def _segments_to_msa_info(segments: List[Dict], duration: float) -> List[Tuple[float, str]]:
    if not segments:
        return [(0.0, "end")]
    items = sorted(segments, key=lambda x: float(x["start"]))
    msa = [(float(items[0]["start"]), str(items[0]["label"]))]
    for seg in items[1:]:
        msa.append((float(seg["start"]), str(seg["label"])))
    last_end = float(items[-1]["end"]) if items else float(duration)
    if last_end < float(duration):
        last_end = float(duration)
    msa.append((last_end, "end"))
    return msa


def _cal_acc(ann_info, est_info, post_digit: int = 3) -> float:
    ann_info_time = [int(round(time_, post_digit) * (10**post_digit)) for time_, _ in ann_info]
    est_info_time = [int(round(time_, post_digit) * (10**post_digit)) for time_, _ in est_info]
    common_start_time = max(ann_info_time[0], est_info_time[0])
    common_end_time = min(ann_info_time[-1], est_info_time[-1])
    time_points = {common_start_time, common_end_time}
    time_points.update({t for t in ann_info_time if common_start_time <= t <= common_end_time})
    time_points.update({t for t in est_info_time if common_start_time <= t <= common_end_time})
    time_points = sorted(time_points)
    total_duration = 0
    total_score = 0
    for idx in range(len(time_points) - 1):
        duration = time_points[idx + 1] - time_points[idx]
        ann_label = ann_info[bisect.bisect_right(ann_info_time, time_points[idx]) - 1][1]
        est_label = est_info[bisect.bisect_right(est_info_time, time_points[idx]) - 1][1]
        total_duration += duration
        if ann_label == est_label:
            total_score += duration
    return total_score / total_duration if total_duration > 0 else 0.0


def _cal_iou(ann_info, est_info) -> float:
    def to_segments(msa):
        return [(msa[i][0], msa[i + 1][0], msa[i][1]) for i in range(len(msa) - 1)]
    ann_segments = to_segments(ann_info)
    est_segments = to_segments(est_info)
    labels = list(set([l for _, _, l in ann_segments] + [l for _, _, l in est_segments]))
    intsec_total = 0.0
    uni_total = 0.0
    for label in labels:
        ann = [(s, e) for s, e, l in ann_segments if l == label]
        est = [(s, e) for s, e, l in est_segments if l == label]
        intersection = 0.0
        for sa, ea in ann:
            for sb, eb in est:
                left = max(sa, sb)
                right = min(ea, eb)
                if left < right:
                    intersection += right - left
        len_a = sum(e - s for s, e in ann)
        len_b = sum(e - s for s, e in est)
        union = len_a + len_b - intersection
        intsec_total += intersection
        uni_total += union
    return float(intsec_total / uni_total) if uni_total > 0 else 0.0


def beat_snap_times(times: List[float], y: np.ndarray, audio_cfg: AudioConfig, max_diff: float = 0.35) -> List[float]:
    if len(times) <= 2:
        return times
    sr = int(getattr(audio_cfg, "sample_rate", 0) or 0)
    hop = int(getattr(audio_cfg, "hop_length", 0) or 0)
    if sr <= 0 or hop <= 0 or y is None or len(y) < sr:
        return times
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    except Exception:
        return times
    if beat_times is None or len(beat_times) == 0:
        return times
    duration = float(times[-1])
    snapped = [0.0]
    for t in times[1:-1]:
        j = int(np.argmin(np.abs(beat_times - t)))
        bt = float(beat_times[j])
        snapped.append(bt if abs(bt - t) <= float(max_diff) else float(t))
    snapped.append(duration)
    snapped = sorted(set(snapped))
    if snapped[0] != 0.0:
        snapped = [0.0] + snapped
    if snapped[-1] != duration:
        snapped.append(duration)
    return snapped


def beat_refine_times(
    times: List[float],
    probs: np.ndarray,
    y: np.ndarray,
    audio_cfg: AudioConfig,
    max_diff: float = 0.35,
    refine_radius_frames: int = 2,
) -> List[float]:
    if len(times) <= 2:
        return times
    sr = int(getattr(audio_cfg, "sample_rate", 0) or 0)
    hop = int(getattr(audio_cfg, "hop_length", 0) or 0)
    if sr <= 0 or hop <= 0 or y is None or len(y) < sr or probs is None or probs.size == 0:
        return times
    try:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
    except Exception:
        return times
    if beat_times is None or len(beat_times) == 0:
        return times
    t_out = int(probs.shape[0])
    dur = float(times[-1])
    refined = [0.0]
    r = int(max(0, refine_radius_frames))
    for t in times[1:-1]:
        t = float(t)
        j = int(np.argmin(np.abs(beat_times - t)))
        bt = float(beat_times[j])
        if abs(bt - t) > float(max_diff):
            refined.append(t)
            continue
        idx = int(round(bt * sr / hop))
        idx = max(0, min(idx, t_out - 1))
        left = max(0, idx - r)
        right = min(t_out - 1, idx + r)
        k = int(left + np.argmax(probs[left : right + 1]))
        refined.append(float(k) * hop / sr)
    refined.append(dur)
    refined = sorted(set(refined))
    if refined[0] != 0.0:
        refined = [0.0] + refined
    if refined[-1] != dur:
        refined.append(dur)
    return refined


def infer_boundaries(
    boundary_bundle: Dict,
    audio_path: str,
    duration_hint: float = 0.0,
    threshold: float = None,
    downsample_factor: int = 3,
    beat_snap: bool = True,
    beat_snap_max: float = 0.35,
    beat_refine_radius_frames: int = 0,
    override_min_distance: int = 0,
    override_max_peaks: int = -1,
    no_hierarchical: bool = False,
    return_probs: bool = False,
) -> Tuple[List[float], float, np.ndarray]:
    model = boundary_bundle["model"]
    audio_cfg = AudioConfig(**boundary_bundle["audio_cfg"])
    boundary_cfg = BoundaryConfig(**boundary_bundle["boundary_cfg"])
    if bool(no_hierarchical):
        boundary_cfg.use_hierarchical = False
    y = None
    if str(audio_path).lower().endswith(".npz"):
        mel = mel_spectrogram(audio_path, audio_cfg)[0]
        duration = float(duration_hint) if float(duration_hint) > 0 else float(mel.shape[1] * audio_cfg.hop_length / audio_cfg.sample_rate)
        beat_snap = False
    else:
        y, _ = librosa.load(audio_path, sr=audio_cfg.sample_rate, mono=True)
        duration = float(len(y) / audio_cfg.sample_rate)
        mel = mel_spectrogram(audio_path, audio_cfg)[0]
    thr = float(boundary_cfg.threshold) if threshold is None else float(threshold)
    min_distance = int(boundary_cfg.min_distance)
    if int(override_min_distance) > 0:
        min_distance = int(override_min_distance)
    max_peaks = int(getattr(boundary_cfg, "max_peaks", 0) or 0)
    if int(override_max_peaks) >= 0:
        max_peaks = int(override_max_peaks)
    if boundary_cfg.use_hierarchical:
        probs = hierarchical_boundary_probs(model, mel, boundary_cfg)
    else:
        use_global = hasattr(model, "local_frames")
        probs = sliding_boundary_probs(model, mel, boundary_cfg, use_global=use_global)
    boundary_cfg.threshold = float(thr)
    if getattr(boundary_cfg, "use_songformer_postprocess", False):
        picks = songformer_pick_boundaries(probs, audio_cfg, boundary_cfg)
    else:
        picks = peak_pick(
            probs,
            float(thr),
            int(min_distance),
            int(getattr(boundary_cfg, "smooth_window", 1)),
            float(getattr(boundary_cfg, "prominence", 0.0)),
            int(max_peaks),
        )
        picks = refine_peaks(probs, picks, radius=int(getattr(boundary_cfg, "refine_radius", 6)))
    edge_margin_sec = float(getattr(boundary_cfg, "edge_margin_sec", 0.0) or 0.0)
    if edge_margin_sec > 0 and float(duration) > 0:
        em = int(round(edge_margin_sec * audio_cfg.sample_rate / audio_cfg.hop_length))
        picks = [int(p) for p in picks if int(p) >= em and int(p) <= int(probs.shape[0]) - 1 - em]
    max_peaks_eff = int(max_peaks)
    if max_peaks_eff <= 0:
        max_peaks_eff = max(4, int(float(duration) / 20.0)) if float(duration) > 0 else 0
    if max_peaks_eff > 0:
        picks = limit_peaks(picks, probs, int(min_distance), int(max_peaks_eff))
    times = [0.0] + [float(p) * audio_cfg.hop_length / audio_cfg.sample_rate for p in picks] + [duration]
    if beat_snap and y is not None:
        if int(beat_refine_radius_frames) > 0:
            times = beat_refine_times(times, probs, y, audio_cfg, max_diff=float(beat_snap_max), refine_radius_frames=int(beat_refine_radius_frames))
        else:
            times = beat_snap_times(times, y, audio_cfg, max_diff=float(beat_snap_max))
    if return_probs:
        return times, duration, probs
    return times, duration, np.asarray([], dtype=np.float32)


def classify_segments_two_stage(
    classifier_bundle: Dict, audio_path: str, segments: List[Dict], audio_cfg: AudioConfig
) -> List[Dict]:
    clf = classifier_bundle["model"]
    clf.eval()
    device = next(clf.parameters()).device
    labels = classifier_bundle["labels"]
    feat_dim = classifier_bundle["feat_dim"]
    input_type = classifier_bundle.get("input_type", "pooled")
    segment_frames = int(classifier_bundle.get("segment_frames", 256))
    mel = mel_spectrogram(audio_path, audio_cfg)[0]
    mel_frames = mel.shape[1]
    n_mels = mel.shape[0]
    out = []
    for seg in segments:
        start = float(seg["start"])
        end = float(seg["end"])
        s = int(start * audio_cfg.sample_rate / audio_cfg.hop_length)
        e = int(end * audio_cfg.sample_rate / audio_cfg.hop_length)
        e = max(e, s + 1)
        if input_type == "mel":
            s = max(0, min(s, mel_frames - 1))
            e = max(s + 1, min(e, mel_frames))
            seg_len = e - s
            target = segment_frames
            if seg_len >= target:
                off = max(0, (seg_len - target) // 2)
                patch = mel[:, s + off : s + off + target]
            else:
                patch = mel[:, s:e]
                pad = target - patch.shape[1]
                patch = np.pad(patch, ((0, 0), (0, pad)), mode="constant")
            x = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = clf(x)
                pred = int(torch.argmax(logits, dim=1).item())
        else:
            s = max(0, min(s, mel_frames - 1))
            e = max(s + 1, min(e, mel_frames))
            seg_mel = mel[:, s:e]
            mean = seg_mel.mean(axis=1)
            std = seg_mel.std(axis=1)
            mx = seg_mel.max(axis=1)
            dur_ratio = (e - s) / max(1, mel_frames)
            center_ratio = ((s + e) / 2) / max(1, mel_frames)
            if feat_dim is None or feat_dim <= n_mels * 3 + 2:
                feat = np.concatenate([mean, std, mx, np.array([dur_ratio, center_ratio], dtype=np.float32)]).astype(np.float32)
            else:
                mn = seg_mel.min(axis=1)
                dur_sec = end - start
                start_ratio = s / max(1, mel_frames)
                end_ratio = e / max(1, mel_frames)
                feat = np.concatenate(
                    [mean, std, mx, mn, np.array([dur_ratio, center_ratio, dur_sec, start_ratio, end_ratio], dtype=np.float32)]
                ).astype(np.float32)
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = clf(x)
                pred = int(torch.argmax(logits, dim=1).item())
        out.append({"start": start, "end": end, "label": str(labels[pred])})
    return out


def evaluate_two_stage(
    data_dir: str,
    boundary_ckpt: str,
    classifier_ckpt: str,
    split: str = "val",
    threshold: float = None,
    beat_snap: bool = True,
    beat_snap_max: float = 0.35,
    beat_refine_radius_frames: int = 0,
    override_min_distance: int = 0,
    override_max_peaks: int = -1,
    downsample_factor: int = 3,
    debug_songs: int = 0,
    no_hierarchical: bool = False,
) -> Dict[str, float]:
    split_res = list_pairs_by_split(data_dir)
    if split_res is None:
        # Fallback to random splitting logic if manifest doesn't exist or doesn't have split info
        # Import here to avoid circular imports if list_pairs is in data.py
        from .data import list_pairs
        all_pairs = list_pairs(data_dir)
        
        # Consistent random shuffle
        import random
        rng = random.Random(42)
        rng.shuffle(all_pairs)
        
        n = len(all_pairs)
        n_test = int(n * 0.1)
        n_val = int(n * 0.1)
        
        # Default splits
        test_pairs = all_pairs[:n_test]
        val_pairs = all_pairs[n_test : n_test + n_val]
        train_pairs = all_pairs[n_test + n_val :]
        
        if split == "train":
            pairs = train_pairs
        elif split == "val":
            pairs = val_pairs
        elif split == "test":
            pairs = test_pairs
        else:
            pairs = val_pairs
    else:
        # Use manifest splits
        if split == "train":
            pairs = split_res[0]
        elif split == "val":
            pairs = split_res[1]
        elif split == "test":
            pairs = split_res[2]
        else:
            pairs = split_res[1]
            
    # Fallback if test is empty but val exists (common in some datasets)
    if not pairs and split == "test":
        print("Warning: No test pairs found. Using validation pairs for evaluation.")
        if split_res is not None:
            pairs = split_res[1]
        elif 'val_pairs' in locals() and val_pairs:
            pairs = val_pairs
            
    if not pairs:
        # Final fallback: if everything fails, use all pairs (or train if all fails)
        # This is just to ensure it runs if data exists but splits are messed up
        if split_res is None and 'all_pairs' in locals() and all_pairs:
             print(f"Warning: No pairs for split={split}, falling back to all_pairs")
             pairs = all_pairs
        else:
             raise ValueError(f"no pairs for split={split}")
    boundary_bundle = load_boundary(boundary_ckpt)
    classifier_bundle = load_classifier(classifier_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    boundary_bundle["model"] = boundary_bundle["model"].to(device)
    classifier_bundle["model"] = classifier_bundle["model"].to(device)
    audio_cfg = AudioConfig(**boundary_bundle["audio_cfg"])

    hr5 = []
    hr3 = []
    accs = []
    acc_seg_list = []
    iou_seg_list = []
    pred_counts = []
    gt_counts = []
    debug_songs = int(max(0, debug_songs))
    debug_lines = []
    for audio_path, ann_path in pairs:
        gt = load_segments(ann_path)
        gt_bounds = [float(s.start) for s in gt[1:]]
        duration_hint = float(gt[-1].end) if gt else 0.0
        times, duration, probs = infer_boundaries(
            boundary_bundle,
            audio_path,
            duration_hint=duration_hint,
            threshold=(None if threshold is None else float(threshold)),
            downsample_factor=int(downsample_factor),
            beat_snap=bool(beat_snap),
            beat_snap_max=float(beat_snap_max),
            beat_refine_radius_frames=int(beat_refine_radius_frames),
            override_min_distance=int(override_min_distance),
            override_max_peaks=int(override_max_peaks),
            no_hierarchical=bool(no_hierarchical),
            return_probs=bool(debug_songs > 0),
        )
        pred_bounds = [float(t) for t in times[1:-1]]
        pred_counts.append(int(len(pred_bounds)))
        gt_counts.append(int(len(gt_bounds)))
        hr5.append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=0.5).get("f_measure", 0.0)))
        hr3.append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=3.0).get("f_measure", 0.0)))
        pred_segments = [{"start": float(times[i]), "end": float(times[i + 1]), "label": "unknown"} for i in range(len(times) - 1)]
        pred_segments = classify_segments_two_stage(classifier_bundle, audio_path, pred_segments, audio_cfg)
        gt_y = gt_frame_labels_7(gt, audio_cfg, duration, int(downsample_factor))
        pred_y = pred_frame_labels_from_segments_7(pred_segments, audio_cfg, duration, int(downsample_factor))
        n = min(int(gt_y.shape[0]), int(pred_y.shape[0]))
        accs.append(framewise_accuracy(pred_y[:n], gt_y[:n], ignore_index=IGNORE_INDEX))
        msa_gt = _segments_to_msa_info([{"start": s.start, "end": s.end, "label": s.label} for s in gt], duration)
        msa_pred = _segments_to_msa_info(pred_segments, duration)
        acc_seg_list.append(_cal_acc(msa_gt, msa_pred))
        iou_seg_list.append(_cal_iou(msa_gt, msa_pred))
        if debug_songs > 0 and len(debug_lines) < debug_songs:
            pmax = float(np.max(probs)) if probs.size else None
            pmean = float(np.mean(probs)) if probs.size else None
            p95 = float(np.quantile(probs, 0.95)) if probs.size else None
            debug_lines.append(
                {
                    "audio": str(audio_path),
                    "duration": float(duration),
                    "gt_bounds": int(len(gt_bounds)),
                    "pred_bounds": int(len(pred_bounds)),
                    "first_pred_bounds": [float(x) for x in pred_bounds[:5]],
                    "first_gt_bounds": [float(x) for x in gt_bounds[:5]],
                    "probs_max": pmax,
                    "probs_p95": p95,
                    "probs_mean": pmean,
                }
            )

    return {
        "ACC": float(np.mean(accs)) if accs else 0.0,
        "ACC_SEG": float(np.mean(acc_seg_list)) if acc_seg_list else 0.0,
        "IOU_SEG": float(np.mean(iou_seg_list)) if iou_seg_list else 0.0,
        "HR.5F": float(np.mean(hr5)) if hr5 else 0.0,
        "HR3F": float(np.mean(hr3)) if hr3 else 0.0,
        "n": int(len(pairs)),
        "pred_bounds_per_song": float(np.mean(pred_counts)) if pred_counts else 0.0,
        "gt_bounds_per_song": float(np.mean(gt_counts)) if gt_counts else 0.0,
        "debug": debug_lines,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--boundary_ckpt", required=True)
    parser.add_argument("--classifier_ckpt", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--sweep_thresholds", default="")
    parser.add_argument("--no_beat_snap", action="store_false", dest="beat_snap")
    parser.set_defaults(beat_snap=True)
    parser.add_argument("--no_hierarchical", action="store_true")
    parser.add_argument("--beat_snap_max", type=float, default=0.35)
    parser.add_argument("--beat_refine_radius_frames", type=int, default=0)
    parser.add_argument("--override_min_distance", type=int, default=0)
    parser.add_argument("--override_max_peaks", type=int, default=-1)
    parser.add_argument("--debug_songs", type=int, default=0)
    parser.add_argument("--downsample_factor", type=int, default=3)
    args = parser.parse_args()
    if str(args.sweep_thresholds).strip():
        thrs = []
        for s in str(args.sweep_thresholds).split(","):
            s = s.strip()
            if not s:
                continue
            thrs.append(float(s))
        best = None
        for thr in thrs:
            res = evaluate_two_stage(
                args.data_dir,
                args.boundary_ckpt,
                args.classifier_ckpt,
                split=args.split,
                threshold=float(thr),
                beat_snap=bool(args.beat_snap),
                beat_snap_max=float(args.beat_snap_max),
                beat_refine_radius_frames=int(args.beat_refine_radius_frames),
                override_min_distance=int(args.override_min_distance),
                override_max_peaks=int(args.override_max_peaks),
                downsample_factor=int(args.downsample_factor),
                debug_songs=int(args.debug_songs),
                no_hierarchical=bool(args.no_hierarchical),
            )
            out = {"threshold": float(thr), **res}
            if best is None or float(out["HR.5F"]) > float(best["HR.5F"]):
                best = out
            print(json.dumps(out, ensure_ascii=False))
        print(json.dumps({"best": best}, ensure_ascii=False, indent=2))
    else:
        res = evaluate_two_stage(
            args.data_dir,
            args.boundary_ckpt,
            args.classifier_ckpt,
            split=args.split,
            threshold=args.threshold,
            beat_snap=bool(args.beat_snap),
            beat_snap_max=float(args.beat_snap_max),
            beat_refine_radius_frames=int(args.beat_refine_radius_frames),
            override_min_distance=int(args.override_min_distance),
            override_max_peaks=int(args.override_max_peaks),
            downsample_factor=int(args.downsample_factor),
            debug_songs=int(args.debug_songs),
            no_hierarchical=bool(args.no_hierarchical),
        )
        print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
