import argparse
import json
import math
import os
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch

from .config import AudioConfig, BoundaryConfig
from .data import mel_spectrogram
from .infer import load_classifier, peak_pick
from .songformer import SongFormerConfig, SongFormerNet


def load_songformer(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["cfg"]
    audio_cfg = cfg.get("audio_cfg") or {}
    sf_cfg = cfg.get("songformer_cfg") or {}
    labels = cfg.get("labels") or []
    model = SongFormerNet(
        n_mels=int(audio_cfg.get("n_mels", AudioConfig.n_mels)),
        num_labels=len(labels),
        cfg=SongFormerConfig(**sf_cfg),
    )
    model.load_state_dict(ckpt["state"])
    model.eval()
    return {
        "model": model,
        "audio_cfg": audio_cfg,
        "songformer_cfg": sf_cfg,
        "labels": labels,
        "max_seconds": float(cfg.get("max_seconds", 420.0)),
        "downsample_factor": int(cfg.get("downsample_factor", sf_cfg.get("downsample_factor", 3))),
    }


def _frame_hop_sec(audio_cfg: AudioConfig, downsample_factor: int) -> float:
    return float(audio_cfg.hop_length * int(downsample_factor) / float(audio_cfg.sample_rate))


def _to_pairs(segments: List[Dict], duration: float) -> List[Tuple[float, str]]:
    pairs: List[Tuple[float, str]] = []
    for seg in segments:
        pairs.append((float(seg["start"]), str(seg["label"])))
    pairs.append((float(duration), "end"))
    return pairs


def _enforce_min_distance(
    times: List[float],
    scores: Optional[List[float]],
    min_distance_sec: float,
    duration: float,
) -> List[float]:
    if not times:
        return [0.0, float(duration)]
    ts = sorted(set(float(t) for t in times if 0.0 < float(t) < float(duration)))
    if min_distance_sec <= 0:
        return [0.0] + ts + [float(duration)]
    if scores is None or len(scores) != len(times):
        kept = [0.0]
        for t in ts:
            if t - kept[-1] >= float(min_distance_sec):
                kept.append(t)
        if float(duration) - kept[-1] <= 1e-6:
            return kept
        return kept + [float(duration)]
    items = [(float(t), float(s)) for t, s in zip(times, scores) if 0.0 < float(t) < float(duration)]
    items.sort(key=lambda x: x[1], reverse=True)
    kept: List[float] = []
    for t, _ in items:
        if all(abs(t - k) >= float(min_distance_sec) for k in kept):
            kept.append(t)
    kept = sorted(kept)
    return [0.0] + kept + [float(duration)]


def beat_align_boundaries(
    raw_times: List[float],
    boundary_probs: np.ndarray,
    audio_y: np.ndarray,
    audio_cfg: AudioConfig,
    downsample_factor: int,
    max_diff: float = 0.35,
    refine_radius_frames: int = 2,
    min_distance_sec: float = 0.0,
    duration: float = 0.0,
) -> List[float]:
    if not raw_times:
        return [0.0, float(duration)]
    tempo, beat_frames = librosa.beat.beat_track(y=audio_y, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
    if beat_times is None or len(beat_times) == 0:
        return _enforce_min_distance(raw_times, None, float(min_distance_sec), float(duration))
    hop = _frame_hop_sec(audio_cfg, downsample_factor)
    t_out = int(boundary_probs.shape[0])
    scores = []
    snapped = []
    for t in raw_times:
        t = float(t)
        if not (0.0 < t < float(duration)):
            continue
        j = int(np.argmin(np.abs(beat_times - t)))
        bt = float(beat_times[j])
        if abs(bt - t) > float(max_diff):
            idx = int(round(t / hop))
            idx = max(0, min(idx, t_out - 1))
            snapped.append(float(idx) * hop)
            scores.append(float(boundary_probs[idx]))
            continue
        b_idx = int(round(bt / hop))
        b_idx = max(0, min(b_idx, t_out - 1))
        r = int(max(0, refine_radius_frames))
        left = max(0, b_idx - r)
        right = min(t_out - 1, b_idx + r)
        k = int(left + np.argmax(boundary_probs[left : right + 1]))
        snapped.append(float(k) * hop)
        scores.append(float(boundary_probs[k]))
    return _enforce_min_distance(snapped, scores, float(min_distance_sec), float(duration))


def boundaries_from_songformer(
    model: torch.nn.Module,
    mel: np.ndarray,
    audio_cfg: AudioConfig,
    downsample_factor: int,
    device: torch.device,
    threshold: float,
    min_distance_frames: int,
    smooth_window: int,
    prominence: float,
    max_peaks: int,
) -> Tuple[np.ndarray, List[float]]:
    mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(device)
    out = model(mel_t, source_id=torch.zeros((1,), device=device, dtype=torch.long))
    bd_logits = out["boundary_logits"].squeeze(0)
    bd_probs = torch.sigmoid(bd_logits).detach().cpu().numpy().astype(np.float32)
    picks = peak_pick(
        bd_probs,
        float(threshold),
        int(min_distance_frames),
        int(smooth_window),
        float(prominence),
        int(max_peaks),
    )
    hop = _frame_hop_sec(audio_cfg, int(downsample_factor))
    times = [float(p) * hop for p in picks]
    return bd_probs, times


def label_segments_two_stage(
    mel: np.ndarray,
    times: List[float],
    audio_cfg: AudioConfig,
    classifier_ckpt: str,
    device: torch.device,
) -> List[Dict]:
    classifier = load_classifier(classifier_ckpt)
    clf = classifier["model"].to(device)
    clf.eval()
    labels = classifier["labels"]
    feat_dim = classifier["feat_dim"]
    input_type = classifier.get("input_type", "pooled")
    segment_frames = int(classifier.get("segment_frames", 256))
    mel_frames = mel.shape[1]
    n_mels = mel.shape[0]

    out = []
    for i in range(len(times) - 1):
        start = float(times[i])
        end = float(times[i + 1])
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
                feat = np.concatenate(
                    [
                        mean,
                        std,
                        mx,
                        np.array([dur_ratio, center_ratio], dtype=np.float32),
                    ]
                ).astype(np.float32)
            else:
                mn = seg_mel.min(axis=1)
                dur_sec = end - start
                start_ratio = s / max(1, mel_frames)
                end_ratio = e / max(1, mel_frames)
                feat = np.concatenate(
                    [
                        mean,
                        std,
                        mx,
                        mn,
                        np.array([dur_ratio, center_ratio, dur_sec, start_ratio, end_ratio], dtype=np.float32),
                    ]
                ).astype(np.float32)
            x = torch.from_numpy(feat).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = clf(x)
                pred = int(torch.argmax(logits, dim=1).item())
        out.append({"start": start, "end": end, "label": str(labels[pred])})
    return out


def label_segments_songformer(
    function_logits: torch.Tensor,
    times: List[float],
    audio_cfg: AudioConfig,
    downsample_factor: int,
    labels: List[str],
    duration: float,
) -> List[Dict]:
    fn_probs = torch.softmax(function_logits, dim=-1).detach().cpu().numpy()
    hop = _frame_hop_sec(audio_cfg, downsample_factor)
    t_out = int(fn_probs.shape[0])
    gt_frames = int(min(t_out, math.ceil(float(duration) / hop)))
    out = []
    for i in range(len(times) - 1):
        start = float(times[i])
        end = float(times[i + 1])
        fs = int(max(0, math.floor(start / hop)))
        fe = int(min(gt_frames, max(fs + 1, math.ceil(end / hop))))
        if fe <= fs:
            fe = min(gt_frames, fs + 1)
        avg = fn_probs[fs:fe].mean(axis=0)
        pred = int(np.argmax(avg))
        out.append({"start": start, "end": end, "label": str(labels[pred])})
    return out


def infer_one(
    audio_path: str,
    songformer_ckpt: str,
    out_prefix: str,
    classifier_ckpt: str = "",
    device: str = "",
    max_seconds: float = 0.0,
    threshold: float = 0.3,
    min_distance_frames: int = 64,
    smooth_window: int = 5,
    prominence: float = 0.0,
    max_peaks: int = 0,
    beat_snap: bool = True,
    beat_snap_max: float = 0.35,
    beat_refine_radius_frames: int = 2,
    min_distance_sec: float = 0.0,
) -> Dict:
    sf = load_songformer(songformer_ckpt)
    audio_cfg = AudioConfig(**sf["audio_cfg"])
    downsample_factor = int(sf["downsample_factor"])
    if max_seconds and float(max_seconds) > 0:
        max_dur = float(max_seconds)
    else:
        max_dur = float(sf.get("max_seconds", 420.0))

    if device:
        dev = torch.device(str(device))
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = sf["model"].to(dev)
    model.eval()

    y, _ = librosa.load(audio_path, sr=audio_cfg.sample_rate, mono=True)
    duration = float(len(y) / audio_cfg.sample_rate)
    duration = min(duration, max_dur)

    mel, _ = mel_spectrogram(audio_path, audio_cfg)
    max_frames = int(math.ceil(max_dur * audio_cfg.sample_rate / audio_cfg.hop_length))
    mel = mel[:, :max_frames]
    if mel.shape[1] < max_frames:
        mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])), mode="constant")

    bd_probs, raw_times = boundaries_from_songformer(
        model,
        mel,
        audio_cfg,
        downsample_factor,
        dev,
        threshold,
        min_distance_frames,
        smooth_window,
        prominence,
        max_peaks,
    )

    times = [0.0] + sorted(set(float(t) for t in raw_times if 0.0 < float(t) < duration)) + [duration]
    if beat_snap and len(times) > 2:
        times = beat_align_boundaries(
            raw_times,
            bd_probs,
            y,
            audio_cfg,
            downsample_factor,
            max_diff=float(beat_snap_max),
            refine_radius_frames=int(beat_refine_radius_frames),
            min_distance_sec=float(min_distance_sec),
            duration=float(duration),
        )
    else:
        times = _enforce_min_distance(raw_times, None, float(min_distance_sec), float(duration))

    segments: List[Dict]
    if classifier_ckpt:
        segments = label_segments_two_stage(mel, times, audio_cfg, classifier_ckpt, dev)
    else:
        mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(dev)
        out = model(mel_t, source_id=torch.zeros((1,), device=dev, dtype=torch.long))
        fn_logits = out["function_logits"].squeeze(0)
        segments = label_segments_songformer(fn_logits, times, audio_cfg, downsample_factor, sf["labels"], duration)

    pairs = _to_pairs(segments, duration)

    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    with open(out_prefix + ".pairs.txt", "w", encoding="utf-8") as f:
        for t, lab in pairs:
            f.write(f"{t:.6f} {lab}\n")
    with open(out_prefix + ".pairs.json", "w", encoding="utf-8") as f:
        json.dump({"pairs": pairs, "duration": duration}, f, ensure_ascii=False, indent=2)
    with open(out_prefix + ".segments.json", "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "duration": duration}, f, ensure_ascii=False, indent=2)
    return {"pairs": pairs, "segments": segments, "duration": duration}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", required=True)
    parser.add_argument("--songformer_ckpt", required=True)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--classifier_ckpt", default="")
    parser.add_argument("--device", default="")
    parser.add_argument("--max_seconds", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--min_distance_frames", type=int, default=64)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--prominence", type=float, default=0.0)
    parser.add_argument("--max_peaks", type=int, default=0)
    parser.add_argument("--beat_snap", action="store_true", default=False)
    parser.add_argument("--beat_snap_max", type=float, default=0.35)
    parser.add_argument("--beat_refine_radius_frames", type=int, default=2)
    parser.add_argument("--min_distance_sec", type=float, default=0.0)
    args = parser.parse_args()

    infer_one(
        args.audio_path,
        args.songformer_ckpt,
        args.out_prefix,
        classifier_ckpt=args.classifier_ckpt,
        device=args.device,
        max_seconds=args.max_seconds,
        threshold=args.threshold,
        min_distance_frames=args.min_distance_frames,
        smooth_window=args.smooth_window,
        prominence=args.prominence,
        max_peaks=args.max_peaks,
        beat_snap=bool(args.beat_snap),
        beat_snap_max=args.beat_snap_max,
        beat_refine_radius_frames=args.beat_refine_radius_frames,
        min_distance_sec=args.min_distance_sec,
    )


if __name__ == "__main__":
    main()
