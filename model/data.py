import json
import os
import csv
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import AudioConfig, BoundaryConfig, functional_labels


@dataclass
class Segment:
    start: float
    end: float
    label: str


def load_segments(path: str) -> List[Segment]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = [Segment(**item) for item in data["segments"]]
    segments.sort(key=lambda s: s.start)
    return segments


def _norm_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


def infer_functional_segments(segments: List[Segment]) -> List[Segment]:
    if not segments:
        return []
    labels = [_norm_label(s.label) for s in segments]
    valid_idx = [i for i, l in enumerate(labels) if l not in {"silence", "end"}]
    if not valid_idx:
        return segments
    func_set = set(functional_labels())
    if sum(1 for i in valid_idx if labels[i] in func_set) >= int(len(valid_idx) * 0.6):
        return segments
    counts: Dict[str, int] = {}
    totals: Dict[str, float] = {}
    first_idx: Dict[str, int] = {}
    for i in valid_idx:
        l = labels[i]
        counts[l] = counts.get(l, 0) + 1
        totals[l] = totals.get(l, 0.0) + (segments[i].end - segments[i].start)
        if l not in first_idx:
            first_idx[l] = i
    repeated = [l for l, c in counts.items() if c >= 2]
    repeated.sort(key=lambda x: (first_idx.get(x, 0), -totals.get(x, 0.0)))
    label_map: Dict[str, str] = {}
    if len(repeated) >= 2:
        label_map[repeated[0]] = "verse"
        label_map[repeated[1]] = "chorus"
        for l in repeated[2:]:
            label_map[l] = "theme"
    elif len(repeated) == 1:
        only = repeated[0]
        first_pos = first_idx.get(only, 0) / max(1, len(valid_idx))
        label_map[only] = "chorus" if first_pos > 0.2 else "verse"

    first_valid = valid_idx[0]
    last_valid = valid_idx[-1]
    chorus_idx = [i for i, l in enumerate(labels) if label_map.get(l) == "chorus"]
    last_chorus = chorus_idx[-1] if chorus_idx else None

    mapped: List[Segment] = []
    for i, seg in enumerate(segments):
        l = labels[i]
        if l in {"silence", "end"}:
            mapped.append(seg)
            continue
        if l in label_map:
            mapped.append(Segment(seg.start, seg.end, label_map[l]))
            continue
        if i == first_valid:
            mapped.append(Segment(seg.start, seg.end, "intro"))
            continue
        if i == last_valid:
            mapped.append(Segment(seg.start, seg.end, "outro"))
            continue
        if last_chorus is not None and i == last_valid - 1 and i > last_chorus:
            mapped.append(Segment(seg.start, seg.end, "coda"))
            continue
        if last_chorus is not None and i > last_chorus:
            mapped.append(Segment(seg.start, seg.end, "outro"))
            continue
        if last_chorus is not None and i < last_chorus:
            mapped.append(Segment(seg.start, seg.end, "pre_chorus"))
            continue
        mapped.append(Segment(seg.start, seg.end, "bridge"))
    return mapped


def list_pairs(data_dir: str) -> List[Tuple[str, str]]:
    manifest = os.path.join(data_dir, "training_manifest.csv")
    if os.path.isfile(manifest):
        pairs: List[Tuple[str, str]] = []
        with open(manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                a = r.get("audio_path", "")
                j = r.get("annotation_path", "")
                if a and j and os.path.exists(a) and os.path.exists(j):
                    pairs.append((a, j))
        return pairs
    audio_dir = os.path.join(data_dir, "audio")
    ann_dir = os.path.join(data_dir, "annotations")
    if not os.path.isdir(audio_dir) or not os.path.isdir(ann_dir):
        return []
    pairs: List[Tuple[str, str]] = []
    for name in os.listdir(audio_dir):
        base, ext = os.path.splitext(name)
        if ext.lower() not in [".wav", ".mp3", ".flac"]:
            continue
        ann_path = os.path.join(ann_dir, f"{base}.json")
        audio_path = os.path.join(audio_dir, name)
        if os.path.exists(ann_path):
            pairs.append((audio_path, ann_path))
    return pairs


def list_pairs_by_split(data_dir: str):
    manifest = os.path.join(data_dir, "training_manifest.csv")
    if not os.path.isfile(manifest):
        return None
    with open(manifest, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "split" not in reader.fieldnames:
            return None
        train_pairs: List[Tuple[str, str]] = []
        val_pairs: List[Tuple[str, str]] = []
        test_pairs: List[Tuple[str, str]] = []
        for r in reader:
            a = r.get("audio_path", "")
            j = r.get("annotation_path", "")
            s = (r.get("split", "") or "").strip().lower()
            if not a or not j or not os.path.exists(a) or not os.path.exists(j):
                continue
            if s == "train":
                train_pairs.append((a, j))
            elif s in {"val", "valid", "validation"}:
                val_pairs.append((a, j))
            elif s == "test":
                test_pairs.append((a, j))
        if not train_pairs and not val_pairs and not test_pairs:
            return None
        return train_pairs, val_pairs, test_pairs


def label_counts(data_dir: str, map_functional: bool = False) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for _, ann_path in list_pairs(data_dir):
        segments = load_segments(ann_path)
        if map_functional:
            segments = infer_functional_segments(segments)
        for seg in segments:
            counts[seg.label] = counts.get(seg.label, 0) + 1
    return counts


def unique_labels(data_dir: str, map_functional: bool = False) -> List[str]:
    labels = sorted(label_counts(data_dir, map_functional=map_functional).keys())
    if "End" in labels:
        labels.remove("End")
    return labels


def mel_spectrogram(path: str, cfg: AudioConfig) -> Tuple[np.ndarray, int]:
    if str(path).lower().endswith(".npz"):
        data = np.load(path, allow_pickle=True)
        mel = data["data"]
        mel = librosa.power_to_db(mel, ref=1.0, amin=1e-10)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
        mel = (mel - float(mel.mean())) / (float(mel.std()) + 1e-6)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        return mel.astype(np.float32), int(cfg.sample_rate)
    y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=2.0,
    )
    ref = float(np.max(mel)) if float(np.max(mel)) > 0 else 1.0
    mel = librosa.power_to_db(mel, ref=ref, amin=1e-10)
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=-80.0)
    mel = (mel - float(mel.mean())) / (float(mel.std()) + 1e-6)
    mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
    return mel.astype(np.float32), sr


def boundary_labels(
    segments: List[Segment],
    n_frames: int,
    cfg: AudioConfig,
    radius_frames: int = 0,
    sigma_frames: float = 0.0,
) -> np.ndarray:
    labels = np.zeros(n_frames, dtype=np.float32)
    if not segments:
        return labels
    radius = int(max(0, radius_frames or 0))
    sigma = float(max(0.0, sigma_frames or 0.0))
    for seg in segments[1:]:
        center = int(seg.start * cfg.sample_rate / cfg.hop_length)
        if not (0 <= center < n_frames):
            continue
        if radius <= 0:
            labels[center] = 1.0
            continue
        left = max(0, center - radius)
        right = min(n_frames - 1, center + radius)
        idx = np.arange(left, right + 1)
        if sigma > 0:
            w = np.exp(-0.5 * ((idx - center) / sigma) ** 2)
        else:
            w = 1.0 - (np.abs(idx - center) / max(1.0, float(radius)))
        labels[left : right + 1] = np.maximum(labels[left : right + 1], w.astype(np.float32))
    labels = np.clip(labels, 0.0, 1.0)
    return labels


def spec_augment(mel: np.ndarray, freq_mask_param: int = 20, time_mask_param: int = 100, num_freq_masks: int = 2, num_time_masks: int = 2) -> np.ndarray:
    mel = mel.copy()
    num_mels, num_frames = mel.shape
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param)
        f0 = np.random.randint(0, max(1, num_mels - f))
        mel[f0 : f0 + f, :] = 0.0

    for _ in range(num_time_masks):
        t = np.random.randint(0, time_mask_param)
        t0 = np.random.randint(0, max(1, num_frames - t))
        mel[:, t0 : t0 + t] = 0.0
    return mel


class BoundaryDataset(Dataset):
    def __init__(self, data_dir: str, audio_cfg: AudioConfig, boundary_cfg: BoundaryConfig, pairs: List[Tuple[str, str]] = None, augment: bool = False):
        self.pairs = pairs if pairs is not None else list_pairs(data_dir)
        self.audio_cfg = audio_cfg
        self.boundary_cfg = boundary_cfg
        self.augment = augment

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, ann_path = self.pairs[idx]
        mel, _ = mel_spectrogram(audio_path, self.audio_cfg)
        segments = load_segments(ann_path)
        labels = boundary_labels(
            segments,
            mel.shape[1],
            self.audio_cfg,
            getattr(self.boundary_cfg, "label_radius", 0),
            getattr(self.boundary_cfg, "label_sigma", 0.0),
        )

        window = self.boundary_cfg.window_frames
        hop = self.boundary_cfg.window_hop
        if mel.shape[1] <= window:
            mel_win = mel
            lab_win = labels
            pad = window - mel.shape[1]
            if pad > 0:
                mel_win = np.pad(mel_win, ((0, 0), (0, pad)), mode="constant")
                lab_win = np.pad(lab_win, (0, pad), mode="constant")
        else:
            boundary_frames = np.where(labels > 0.5)[0]
            # Balanced Sampling: 60% centered on boundary, 40% random background
            # This prevents overfitting to boundaries and reduces false positives
            if len(boundary_frames) > 0 and np.random.rand() < 0.6:
                center = int(np.random.choice(boundary_frames))
                low = max(0, center - window + 1)
                high = min(center, mel.shape[1] - window)
                if high >= low:
                    start = int(np.random.randint(low, high + 1))
                else:
                    start = max(0, min(center - window // 2, mel.shape[1] - window))
            else:
                start = np.random.randint(0, max(1, mel.shape[1] - window))
            mel_win = mel[:, start : start + window]
            lab_win = labels[start : start + window]

        if self.augment and np.random.rand() < 0.5:
            mel_win = spec_augment(mel_win)

        return {
            "mel": torch.from_numpy(mel_win),
            "labels": torch.from_numpy(lab_win),
        }


class MultiResolutionBoundaryDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        audio_cfg: AudioConfig,
        boundary_cfg: BoundaryConfig,
        pairs: List[Tuple[str, str]] = None,
        local_window_sec: float = 30.0,
        global_window_sec: float = 420.0,
    ):
        self.pairs = pairs if pairs is not None else list_pairs(data_dir)
        self.audio_cfg = audio_cfg
        self.boundary_cfg = boundary_cfg
        self.local_frames = int(local_window_sec * audio_cfg.sample_rate / audio_cfg.hop_length)
        self.global_frames = int(global_window_sec * audio_cfg.sample_rate / audio_cfg.hop_length)
        self._mel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mel_cache_max = 16

    def _get_mel(self, audio_path: str) -> np.ndarray:
        cached = self._mel_cache.get(audio_path)
        if cached is not None:
            self._mel_cache.move_to_end(audio_path)
            return cached
        mel, _ = mel_spectrogram(audio_path, self.audio_cfg)
        self._mel_cache[audio_path] = mel
        self._mel_cache.move_to_end(audio_path)
        while len(self._mel_cache) > self._mel_cache_max:
            self._mel_cache.popitem(last=False)
        return mel

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, ann_path = self.pairs[idx]
        mel = self._get_mel(audio_path)
        segments = load_segments(ann_path)
        labels = boundary_labels(
            segments,
            mel.shape[1],
            self.audio_cfg,
            getattr(self.boundary_cfg, "label_radius", 0),
            getattr(self.boundary_cfg, "label_sigma", 0.0),
        )
        total_frames = mel.shape[1]
        if total_frames > self.global_frames:
            mel = mel[:, : self.global_frames]
            labels = labels[: self.global_frames]
        mel_global = mel.copy()
        if mel.shape[1] > self.local_frames:
            factor = max(1, mel.shape[1] // self.local_frames)
            mel_global_down = mel[:, ::factor]
            pad = self.local_frames - mel_global_down.shape[1]
            if pad > 0:
                mel_global_down = np.pad(mel_global_down, ((0, 0), (0, pad)), mode="constant")
        else:
            mel_global_down = mel
            pad = self.local_frames - mel.shape[1]
            if pad > 0:
                mel_global_down = np.pad(mel_global_down, ((0, 0), (0, pad)), mode="constant")
        return {
            "mel": torch.from_numpy(mel.astype(np.float32)),
            "mel_global": torch.from_numpy(mel_global_down.astype(np.float32)),
            "labels": torch.from_numpy(labels.astype(np.float32)),
        }


class ClassifierDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        audio_cfg: AudioConfig,
        label_to_id: Dict[str, int],
        pairs: List[Tuple[str, str]] = None,
        min_seg_seconds: float = 0.5,
        max_items: int = None,
        seed: int = 42,
        map_functional: bool = False,
    ):
        self.pairs = pairs if pairs is not None else list_pairs(data_dir)
        self.audio_cfg = audio_cfg
        self.label_to_id = label_to_id
        self.num_classes = len(label_to_id)
        self.items: List[Tuple[str, Segment]] = []
        for audio_path, ann_path in self.pairs:
            try:
                segments = load_segments(ann_path)
            except Exception:
                continue
            if map_functional:
                segments = infer_functional_segments(segments)
            for seg in segments:
                if seg.label in self.label_to_id and (seg.end - seg.start) >= min_seg_seconds:
                    self.items.append((audio_path, seg))
        if not self.items:
            for audio_path, ann_path in self.pairs:
                try:
                    segments = load_segments(ann_path)
                except Exception:
                    continue
                if map_functional:
                    segments = infer_functional_segments(segments)
                if segments:
                    self.items.append((audio_path, segments[0]))
        if max_items is not None and len(self.items) > max_items:
            rng = random.Random(seed)
            self.items = rng.sample(self.items, max_items)
        self._mel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mel_cache_max = 8

    def _get_mel(self, audio_path: str) -> np.ndarray:
        cached = self._mel_cache.get(audio_path)
        if cached is not None:
            self._mel_cache.move_to_end(audio_path)
            return cached
        mel, _ = mel_spectrogram(audio_path, self.audio_cfg)
        self._mel_cache[audio_path] = mel
        self._mel_cache.move_to_end(audio_path)
        while len(self._mel_cache) > self._mel_cache_max:
            self._mel_cache.popitem(last=False)
        return mel

    def __len__(self) -> int:
        return len(self.items)

    def sample_weights(self) -> torch.Tensor:
        labels = [self.label_to_id.get(seg.label, 0) for _, seg in self.items]
        if not labels:
            return torch.ones((1,), dtype=torch.float32)
        counts = np.bincount(labels, minlength=self.num_classes)
        counts = np.maximum(counts, 1)
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.mean()
        return torch.tensor([weights[l] for l in labels], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, seg = self.items[idx]
        mel = self._get_mel(audio_path)
        mel_frames = mel.shape[1]
        start = int(seg.start * self.audio_cfg.sample_rate / self.audio_cfg.hop_length)
        end = int(seg.end * self.audio_cfg.sample_rate / self.audio_cfg.hop_length)
        if mel_frames <= 1:
            pooled = np.zeros((mel.shape[0] * 4 + 5,), dtype=np.float32)
            label_id = self.label_to_id.get(seg.label, 0)
            return {
                "feat": torch.from_numpy(pooled),
                "label": torch.tensor(label_id, dtype=torch.long),
            }
        start = max(0, min(start, mel_frames - 1))
        end = max(start + 1, min(end, mel_frames))
        seg_mel = mel[:, start:end]
        mean = seg_mel.mean(axis=1)
        std = seg_mel.std(axis=1)
        mx = seg_mel.max(axis=1)
        mn = seg_mel.min(axis=1)
        dur_ratio = (end - start) / max(1, mel_frames)
        center_ratio = ((start + end) / 2) / max(1, mel_frames)
        dur_sec = seg.end - seg.start
        start_ratio = start / max(1, mel_frames)
        end_ratio = end / max(1, mel_frames)
        pooled = np.concatenate(
            [
                mean,
                std,
                mx,
                mn,
                np.array([dur_ratio, center_ratio, dur_sec, start_ratio, end_ratio], dtype=np.float32),
            ]
        ).astype(np.float32)
        if not np.all(np.isfinite(pooled)):
            pooled = np.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
        label_id = self.label_to_id.get(seg.label, 0)
        return {
            "feat": torch.from_numpy(pooled),
            "label": torch.tensor(label_id, dtype=torch.long),
        }


class ClassifierMelDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        audio_cfg: AudioConfig,
        label_to_id: Dict[str, int],
        pairs: List[Tuple[str, str]] = None,
        min_seg_seconds: float = 0.5,
        max_items: int = None,
        seed: int = 42,
        map_functional: bool = False,
        segment_frames: int = 256,
        train: bool = True,
    ):
        self.pairs = pairs if pairs is not None else list_pairs(data_dir)
        self.audio_cfg = audio_cfg
        self.label_to_id = label_to_id
        self.items: List[Tuple[str, Segment]] = []
        for audio_path, ann_path in self.pairs:
            try:
                segments = load_segments(ann_path)
            except Exception:
                continue
            if map_functional:
                segments = infer_functional_segments(segments)
            for seg in segments:
                if seg.label in self.label_to_id and (seg.end - seg.start) >= min_seg_seconds:
                    self.items.append((audio_path, seg))
        if max_items is not None and len(self.items) > max_items:
            rng = random.Random(seed)
            self.items = rng.sample(self.items, max_items)
        self.segment_frames = int(segment_frames)
        self.train = bool(train)
        self.rng = random.Random(seed)
        self._mel_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mel_cache_max = 8

    def _get_mel(self, audio_path: str) -> np.ndarray:
        cached = self._mel_cache.get(audio_path)
        if cached is not None:
            self._mel_cache.move_to_end(audio_path)
            return cached
        mel, _ = mel_spectrogram(audio_path, self.audio_cfg)
        self._mel_cache[audio_path] = mel
        self._mel_cache.move_to_end(audio_path)
        while len(self._mel_cache) > self._mel_cache_max:
            self._mel_cache.popitem(last=False)
        return mel

    def __len__(self) -> int:
        return len(self.items)

    def sample_weights(self) -> torch.Tensor:
        labels = [self.label_to_id.get(seg.label, 0) for _, seg in self.items]
        if not labels:
            return torch.ones((1,), dtype=torch.float32)
        counts = np.bincount(labels, minlength=len(self.label_to_id))
        counts = np.maximum(counts, 1)
        weights = 1.0 / np.sqrt(counts)
        weights = weights / weights.mean()
        return torch.tensor([weights[l] for l in labels], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, seg = self.items[idx]
        mel = self._get_mel(audio_path)
        mel_frames = mel.shape[1]
        start = int(seg.start * self.audio_cfg.sample_rate / self.audio_cfg.hop_length)
        end = int(seg.end * self.audio_cfg.sample_rate / self.audio_cfg.hop_length)
        if mel_frames <= 1:
            patch = np.zeros((mel.shape[0], self.segment_frames), dtype=np.float32)
            label_id = self.label_to_id.get(seg.label, 0)
            return {"mel": torch.from_numpy(patch), "label": torch.tensor(label_id, dtype=torch.long)}
        start = max(0, min(start, mel_frames - 1))
        end = max(start + 1, min(end, mel_frames))
        seg_len = end - start
        target = self.segment_frames
        if seg_len >= target:
            if self.train:
                off = self.rng.randint(0, seg_len - target)
            else:
                off = max(0, (seg_len - target) // 2)
            s = start + off
            e = s + target
            patch = mel[:, s:e]
        else:
            patch = mel[:, start:end]
            pad = target - patch.shape[1]
            left = 0
            right = pad
            if self.train and pad > 0:
                left = self.rng.randint(0, pad)
                right = pad - left
            patch = np.pad(patch, ((0, 0), (left, right)), mode="constant")
        if self.train and np.random.rand() < 0.5:
            patch = spec_augment(patch, freq_mask_param=10, time_mask_param=20)
        if not np.all(np.isfinite(patch)):
            patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)
        label_id = self.label_to_id.get(seg.label, 0)
        return {"mel": torch.from_numpy(patch.astype(np.float32)), "label": torch.tensor(label_id, dtype=torch.long)}
