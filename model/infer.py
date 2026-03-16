import os
from typing import Dict, List

import librosa
import numpy as np
import torch

from .config import AudioConfig, BoundaryConfig, TransformerConfig
from .model import BoundaryNet, MultiScaleTransformerBoundaryNet, MultiScaleTransformerBoundaryNetV2, MultiScaleTransformerBoundaryNetV3, MultiResolutionBoundaryNet, SegmentClassifier, SegmentMelCNN, SegmentMelAttn, SegmentMelAttnGate, TransformerBoundaryNet


def load_boundary(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg", ckpt.get("config"))
    state = ckpt.get("state", ckpt.get("model_state_dict"))
    arch = cfg.get("arch", {"type": "transformer"})
    if isinstance(arch, str):
        arch = {"type": arch}
    
    if arch.get("type") == "ms_transformer":
        tcfg = TransformerConfig(**arch.get("tcfg", {}))
        model = MultiScaleTransformerBoundaryNet(
            n_mels=cfg["audio_cfg"]["n_mels"],
            d_model=int(arch.get("d_model", 128)),
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            long_pool=int(arch.get("long_pool", 4)),
        )
    elif arch.get("type") == "ms_transformer_v2":
        tcfg = TransformerConfig(**arch.get("tcfg", {}))
        model = MultiScaleTransformerBoundaryNetV2(
            n_mels=cfg["audio_cfg"]["n_mels"],
            d_model=int(arch.get("d_model", 192)),
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            long_pool=int(arch.get("long_pool", 4)),
        )
    elif arch.get("type") == "songformer_ds":
        tcfg = TransformerConfig(**arch.get("tcfg", {}))
        model = MultiScaleTransformerBoundaryNetV3(
            n_mels=cfg["audio_cfg"]["n_mels"],
            d_model=int(arch.get("d_model", 192)),
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            downsample_kernel=int(arch.get("downsample_kernel", 3)),
            downsample_stride=int(arch.get("downsample_stride", 3)),
            downsample_dropout=float(arch.get("downsample_dropout", 0.1)),
            drop_path_rate=float(arch.get("drop_path_rate", 0.0)), # Add drop_path_rate
        )
    elif arch.get("type") == "multi_res":
        tcfg = TransformerConfig(**arch.get("tcfg", {}))
        model = MultiResolutionBoundaryNet(
            n_mels=cfg["audio_cfg"]["n_mels"],
            d_model=int(arch.get("d_model", 256)),
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
            hop_length=cfg["audio_cfg"]["hop_length"],
            sample_rate=cfg["audio_cfg"]["sample_rate"],
        )
    elif arch.get("type") == "transformer":
        tcfg = TransformerConfig(**arch.get("tcfg", {}))
        model = TransformerBoundaryNet(
            n_mels=cfg["audio_cfg"]["n_mels"],
            d_model=128,
            nhead=tcfg.nhead,
            num_layers=tcfg.num_layers,
            dim_feedforward=tcfg.dim_feedforward,
            dropout=tcfg.dropout,
        )
    else:
        model = BoundaryNet(cfg["audio_cfg"]["n_mels"])
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "audio_cfg": cfg["audio_cfg"], "boundary_cfg": cfg["boundary_cfg"]}


def load_classifier(path: str) -> Dict:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("cfg", ckpt.get("config"))
    state = ckpt.get("state", ckpt.get("model_state_dict"))
    input_type = cfg.get("input_type", "pooled")
    arch = cfg.get("arch", None)
    segment_frames = int(cfg.get("segment_frames", 256))
    feat_dim = cfg.get("feat_dim", cfg["audio_cfg"]["n_mels"] * 3 + 2)
    hidden = cfg.get("hidden", 256)
    if input_type == "mel":
        if arch == "mel_attn":
            channels = int(cfg.get("channels", max(16, hidden // 16)))
            attn_dim = int(cfg.get("attn_dim", max(64, hidden // 6)))
            attn_heads = int(cfg.get("attn_heads", 4))
            model = SegmentMelAttn(cfg["audio_cfg"]["n_mels"], cfg["labels"], channels=channels, attn_dim=attn_dim, heads=attn_heads)
        elif arch in {"mel_attn_gate", "mel_attn_gate_ms"}:
            channels = int(cfg.get("channels", max(16, hidden // 16)))
            attn_dim = int(cfg.get("attn_dim", max(64, hidden // 6)))
            attn_heads = int(cfg.get("attn_heads", 4))
            model = SegmentMelAttnGate(cfg["audio_cfg"]["n_mels"], cfg["labels"], channels=channels, attn_dim=attn_dim, heads=attn_heads)
        else:
            model = SegmentMelCNN(cfg["audio_cfg"]["n_mels"], cfg["labels"], channels=max(16, hidden // 16))
        feat_dim = None
    else:
        model = SegmentClassifier(feat_dim, cfg["labels"], hidden)
    
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # Robust handling for 1536 vs 192 dimension mismatch in SegmentMelAttnGate
        if "size mismatch" in str(e) and arch in {"mel_attn_gate", "mel_attn_gate_ms"}:
            print(f"Warning: Classifier size mismatch detected ({e}). Attempting to load with channels=768 (High-Dim Config)...")
            try:
                # Re-instantiate with channels=768 (d_model=1536) and heads=8
                # The size mismatch indicates d_model=1536 (channels=768) and head input 12288 (1536*8)
                # Also attn_dim mismatch (384 vs 128)
                attn_dim = 384
                attn_heads = 8 # Force 8 heads for 1536 dim model
                print(f"DEBUG: Retrying SegmentMelAttnGate with channels=768, attn_dim={attn_dim}, heads={attn_heads}")
                model = SegmentMelAttnGate(
                    cfg["audio_cfg"]["n_mels"], 
                    cfg["labels"], 
                    channels=768, 
                    attn_dim=attn_dim, 
                    heads=attn_heads
                )
                model.load_state_dict(state)
                print("Successfully loaded classifier with corrected dimensions (channels=768, heads=8).")
            except Exception as e2:
                print(f"Failed to recover from size mismatch: {e2}")
                raise e
        else:
            raise e
            
    model.eval()
    return {
        "model": model,
        "audio_cfg": cfg["audio_cfg"],
        "labels": cfg["labels"],
        "feat_dim": feat_dim,
        "input_type": input_type,
        "segment_frames": segment_frames,
    }


def compute_mel(y: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
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
    return mel.astype(np.float32)


def _local_maxima_torch(tensor: torch.Tensor, filter_size: int = 3) -> torch.Tensor:
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    pad = filter_size // 2
    padded = torch.nn.functional.pad(tensor, (pad, pad), mode="constant", value=-torch.inf)
    rolling = padded.unfold(1, filter_size, 1)
    center = filter_size // 2
    mask = torch.eq(rolling[:, :, center], torch.max(rolling, dim=-1).values)
    out = torch.zeros_like(tensor)
    out[mask] = tensor[mask]
    return out.squeeze(0)


def _songformer_peak_picking(boundary_activation: np.ndarray, window_past: int, window_future: int) -> np.ndarray:
    window_size = int(window_past + window_future)
    window_size = window_size + 1
    boundary_activation_padded = np.pad(boundary_activation, (window_past, window_future), mode="constant")
    max_filter = np.lib.stride_tricks.sliding_window_view(boundary_activation_padded, window_size)
    local_maxima = (boundary_activation == np.max(max_filter, axis=-1)) & (boundary_activation > 0)
    past_window = np.lib.stride_tricks.sliding_window_view(
        boundary_activation_padded[: -(window_future + 1)], window_past
    )
    future_window = np.lib.stride_tricks.sliding_window_view(
        boundary_activation_padded[window_past + 1 :], window_future
    )
    past_mean = np.mean(past_window, axis=-1)
    future_mean = np.mean(future_window, axis=-1)
    strength = boundary_activation - ((past_mean + future_mean) / 2)
    candidates = np.flatnonzero(local_maxima)
    strength_values = strength[candidates]
    strength_act = np.zeros_like(boundary_activation)
    strength_act[candidates] = strength_values
    return strength_act


def songformer_pick_boundaries(
    probs: np.ndarray, audio_cfg: AudioConfig, boundary_cfg: BoundaryConfig
) -> List[int]:
    thr = float(getattr(boundary_cfg, "threshold", 0.0))
    ds = int(max(1, getattr(boundary_cfg, "postprocess_downsample_factor", 3)))
    probs_ds = probs[::ds]
    filter_size = int(getattr(boundary_cfg, "local_maxima_filter_size", 3))
    if filter_size % 2 == 0:
        filter_size += 1
    prob_sections = _local_maxima_torch(torch.from_numpy(probs_ds.astype(np.float32)), filter_size=filter_size)
    prob_sections = prob_sections.cpu().numpy()
    frame_rates = float(audio_cfg.sample_rate / audio_cfg.hop_length) / float(ds)
    window_past = int(getattr(boundary_cfg, "postprocess_window_past_sec", 12.0) * frame_rates)
    window_future = int(getattr(boundary_cfg, "postprocess_window_future_sec", 12.0) * frame_rates)
    window_past = max(1, window_past)
    window_future = max(1, window_future)
    strength = _songformer_peak_picking(prob_sections, window_past=window_past, window_future=window_future)
    if thr > 0:
        picks_ds = np.flatnonzero(strength >= thr)
    else:
        picks_ds = np.flatnonzero(strength > 0.0)
    picks = [int(p) * ds for p in picks_ds]
    if not picks and thr > 0:
        picks = peak_pick(
            probs,
            float(thr),
            int(getattr(boundary_cfg, "min_distance", 1)),
            int(getattr(boundary_cfg, "smooth_window", 1)),
            float(getattr(boundary_cfg, "prominence", 0.0)),
            int(getattr(boundary_cfg, "max_peaks", 0) or 0),
        )
    if picks:
        picks = refine_peaks(probs, picks, radius=int(getattr(boundary_cfg, "refine_radius", 6)))
    return picks


def sliding_boundary_probs(model, mel: np.ndarray, cfg: BoundaryConfig, use_global: bool = False) -> np.ndarray:
    device = next(model.parameters()).device
    n_frames = mel.shape[1]
    probs = np.zeros(n_frames, dtype=np.float32)
    if use_global and hasattr(model, "local_frames"):
        local_frames = model.local_frames
        global_frames = model.global_frames
        mel_trunc = mel[:, :global_frames] if n_frames > global_frames else mel
        factor = max(1, mel_trunc.shape[1] // local_frames)
        mel_global = mel_trunc[:, ::factor]
        pad = local_frames - mel_global.shape[1]
        if pad > 0:
            mel_global = np.pad(mel_global, ((0, 0), (0, pad)), mode="constant")
        with torch.no_grad():
            x = torch.from_numpy(mel_trunc).unsqueeze(0).float().to(device)
            xg = torch.from_numpy(mel_global).unsqueeze(0).float().to(device)
            logits = model(x, xg)
            p = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        probs[: len(p)] = p
        return probs
    window = cfg.window_frames
    hop = cfg.window_hop
    counts = np.zeros(n_frames, dtype=np.float32)
    if n_frames <= window:
        with torch.no_grad():
            x = torch.from_numpy(mel).unsqueeze(0).float().to(device)
            logits = model(x)
            p = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
        probs[: n_frames] = p
        counts[: n_frames] = 1
        return probs

    offsets = getattr(cfg, "window_offsets", (0,))
    if offsets is None:
        offsets = (0,)
    for off in offsets:
        off = int(off)
        if off < 0 or off >= hop:
            continue
        start = off
        while start + window <= n_frames:
            chunk = mel[:, start : start + window]
            with torch.no_grad():
                x = torch.from_numpy(chunk).unsqueeze(0).float().to(device)
                logits = model(x)
                p = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            probs[start : start + window] += p
            counts[start : start + window] += 1
            start += hop
    counts[counts == 0] = 1
    return probs / counts


def hierarchical_boundary_probs(model, mel: np.ndarray, cfg: BoundaryConfig) -> np.ndarray:
    n_frames = mel.shape[1]
    if n_frames <= cfg.window_frames:
        return sliding_boundary_probs(model, mel, cfg, use_global=False)

    factor = max(1, int(cfg.window_frames / cfg.coarse_window_frames))
    coarse_len = n_frames // factor
    if coarse_len <= 1:
        return sliding_boundary_probs(model, mel, cfg, use_global=False)
    trimmed = mel[:, : coarse_len * factor]
    coarse = trimmed.reshape(mel.shape[0], coarse_len, factor).mean(axis=2)
    coarse_cfg = BoundaryConfig(
        window_frames=cfg.coarse_window_frames,
        window_hop=cfg.coarse_hop,
        window_offsets=getattr(cfg, "window_offsets", (0,)),
        threshold=min(float(cfg.threshold), 0.05),
        min_distance=max(1, cfg.min_distance // factor),
        use_hierarchical=False,
    )
    coarse_probs = sliding_boundary_probs(model, coarse, coarse_cfg, use_global=False)
    coarse_picks = peak_pick(
        coarse_probs,
        float(coarse_cfg.threshold),
        int(coarse_cfg.min_distance),
        int(getattr(cfg, "smooth_window", 1)),
        float(getattr(cfg, "prominence", 0.0)),
        0,
    )
    if len(coarse_picks) < 4:
        return sliding_boundary_probs(model, mel, cfg, use_global=False)
    candidates = set()
    for p in coarse_picks:
        center = p * factor
        for t in range(max(0, center - cfg.refine_radius), min(n_frames, center + cfg.refine_radius + 1)):
            candidates.add(t)

    probs = np.zeros(n_frames, dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)
    window = cfg.window_frames
    hop = cfg.window_hop
    offsets = getattr(cfg, "window_offsets", (0,))
    if offsets is None:
        offsets = (0,)
    for off in offsets:
        off = int(off)
        if off < 0 or off >= hop:
            continue
        start = off
        while start + window <= n_frames:
            mid = start + window // 2
            if candidates and mid not in candidates:
                start += hop
                continue
            chunk = mel[:, start : start + window]
            with torch.no_grad():
                x = torch.from_numpy(chunk).unsqueeze(0).to(next(model.parameters()).device)
                logits = model(x)
                p = torch.sigmoid(logits).squeeze(0).detach().cpu().numpy()
            probs[start : start + window] += p
            counts[start : start + window] += 1
            start += hop
    counts[counts == 0] = 1
    return probs / counts


def peak_pick(
    probs: np.ndarray,
    threshold: float,
    min_distance: int,
    smooth_window: int = 1,
    prominence: float = 0.0,
    max_peaks: int = 0,
) -> List[int]:
    if smooth_window is not None and smooth_window > 1:
        w = int(smooth_window)
        kernel = np.ones(w, dtype=np.float32) / float(w)
        probs = np.convolve(probs.astype(np.float32), kernel, mode="same")
    picks = []
    last = -min_distance
    for i in range(1, len(probs) - 1):
        if probs[i] < threshold:
            continue
        if probs[i] < probs[i - 1] or probs[i] < probs[i + 1]:
            continue
        if (probs[i] - max(probs[i - 1], probs[i + 1])) < prominence:
            continue
        if i - last >= min_distance:
            picks.append(i)
            last = i
    if max_peaks is not None and int(max_peaks) > 0 and len(picks) > int(max_peaks):
        order = sorted(picks, key=lambda k: float(probs[k]), reverse=True)
        kept = []
        for k in order:
            if len(kept) >= int(max_peaks):
                break
            if all(abs(k - j) >= min_distance for j in kept):
                kept.append(k)
        picks = sorted(kept)
    return picks


def limit_peaks(
    picks: List[int],
    probs: np.ndarray,
    min_distance: int,
    max_peaks: int,
) -> List[int]:
    max_peaks = int(max_peaks)
    if max_peaks <= 0 or len(picks) <= max_peaks:
        return picks
    order = sorted(picks, key=lambda k: float(probs[k]), reverse=True)
    kept = []
    for k in order:
        if len(kept) >= max_peaks:
            break
        if all(abs(int(k) - int(j)) >= int(min_distance) for j in kept):
            kept.append(int(k))
    kept.sort()
    return kept


def refine_peaks(probs: np.ndarray, picks: List[int], radius: int = 6) -> List[int]:
    if probs is None or len(probs) == 0 or not picks:
        return picks
    r = int(max(0, radius))
    if r == 0:
        return sorted(set(int(p) for p in picks))
    out = []
    n = len(probs)
    for p in picks:
        p = int(p)
        left = max(0, p - r)
        right = min(n - 1, p + r)
        k = int(left + np.argmax(probs[left : right + 1]))
        out.append(k)
    return sorted(set(out))


def beat_guided_peak_pick(
    probs: np.ndarray,
    y: np.ndarray,
    audio_cfg: AudioConfig,
    cfg: BoundaryConfig,
    threshold: float,
) -> List[int]:
    if y is None or probs is None or len(probs) == 0:
        return peak_pick(
            probs,
            float(threshold),
            int(cfg.min_distance),
            int(getattr(cfg, "smooth_window", 1)),
            float(getattr(cfg, "prominence", 0.0)),
            int(getattr(cfg, "max_peaks", 0) or 0),
        )
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
    if beat_frames is None or len(beat_frames) == 0:
        return peak_pick(
            probs,
            float(threshold),
            int(cfg.min_distance),
            int(getattr(cfg, "smooth_window", 1)),
            float(getattr(cfg, "prominence", 0.0)),
            int(getattr(cfg, "max_peaks", 0) or 0),
        )
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    beat_frames = beat_frames[(beat_frames > 0) & (beat_frames < len(probs) - 1)]
    if beat_frames.size == 0:
        return []
    r = 2
    candidates = []
    for b in beat_frames:
        left = max(1, int(b) - r)
        right = min(len(probs) - 2, int(b) + r)
        k = int(left + np.argmax(probs[left : right + 1]))
        candidates.append(k)
    candidates = sorted(set(candidates))
    scores = [(k, float(probs[k])) for k in candidates if float(probs[k]) >= float(threshold)]
    if not scores:
        return []
    scores.sort(key=lambda x: x[1], reverse=True)
    min_distance = int(cfg.min_distance)
    duration = float(len(y) / float(audio_cfg.sample_rate))
    max_peaks = int(getattr(cfg, "max_peaks", 0) or 0)
    if max_peaks <= 0:
        max_peaks = max(4, int(duration / 22.0))
    kept = []
    for k, _s in scores:
        if len(kept) >= int(max_peaks):
            break
        if all(abs(int(k) - int(j)) >= int(min_distance) for j in kept):
            kept.append(int(k))
    kept.sort()
    return kept


def analyze(audio_path: str, boundary_path: str, classifier_path: str) -> Dict:
    if not os.path.exists(boundary_path) or not os.path.exists(classifier_path):
        raise FileNotFoundError("missing checkpoints")

    boundary = load_boundary(boundary_path)
    classifier = load_classifier(classifier_path)
    audio_cfg = AudioConfig(**boundary["audio_cfg"])
    boundary_cfg = BoundaryConfig(**boundary["boundary_cfg"])

    y, _ = librosa.load(audio_path, sr=audio_cfg.sample_rate, mono=True)
    mel = compute_mel(y, audio_cfg)
    if boundary_cfg.use_hierarchical:
        probs = hierarchical_boundary_probs(boundary["model"], mel, boundary_cfg)
    else:
        probs = sliding_boundary_probs(boundary["model"], mel, boundary_cfg)
    if getattr(boundary_cfg, "use_songformer_postprocess", False):
        picks = songformer_pick_boundaries(probs, audio_cfg, boundary_cfg)
    elif getattr(boundary_cfg, "beat_snap", False):
        picks = beat_guided_peak_pick(probs, y, audio_cfg, boundary_cfg, float(boundary_cfg.threshold))
    else:
        picks = peak_pick(
            probs,
            boundary_cfg.threshold,
            boundary_cfg.min_distance,
            getattr(boundary_cfg, "smooth_window", 1),
            getattr(boundary_cfg, "prominence", 0.0),
            getattr(boundary_cfg, "max_peaks", 0),
        )
    times = [0.0] + [p * audio_cfg.hop_length / audio_cfg.sample_rate for p in picks]
    duration = len(y) / audio_cfg.sample_rate
    times.append(duration)
    if getattr(boundary_cfg, "beat_snap", False) and len(times) > 2:
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
        beat_times = librosa.frames_to_time(beat_frames, sr=audio_cfg.sample_rate, hop_length=audio_cfg.hop_length)
        if beat_times is not None and len(beat_times) > 0:
            snapped = [0.0]
            max_diff = float(getattr(boundary_cfg, "beat_snap_max", 0.35))
            for t in times[1:-1]:
                j = int(np.argmin(np.abs(beat_times - t)))
                bt = float(beat_times[j])
                snapped.append(bt if abs(bt - t) <= max_diff else float(t))
            snapped.append(duration)
            snapped = sorted(set(snapped))
            if snapped[0] != 0.0:
                snapped = [0.0] + snapped
            if snapped[-1] != duration:
                snapped.append(duration)
            times = snapped

    segments = []
    labels = classifier["labels"]
    clf = classifier["model"]
    feat_dim = classifier["feat_dim"]
    input_type = classifier.get("input_type", "pooled")
    segment_frames = int(classifier.get("segment_frames", 256))
    n_mels = mel.shape[0]
    for i in range(len(times) - 1):
        start = times[i]
        end = times[i + 1]
        s = int(start * audio_cfg.sample_rate / audio_cfg.hop_length)
        e = int(end * audio_cfg.sample_rate / audio_cfg.hop_length)
        e = max(e, s + 1)
        mel_frames = mel.shape[1]
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
            x = torch.from_numpy(patch.astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = clf(x)
                pred = int(torch.argmax(logits, dim=1).item())
        else:
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
            with torch.no_grad():
                logits = clf(torch.from_numpy(feat).unsqueeze(0))
                pred = int(torch.argmax(logits, dim=1).item())
        segments.append({"start": float(start), "end": float(end), "label": labels[pred]})

    return {"segments": segments, "duration": duration}
