import argparse
import json
import math
import os
import random
from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .config import AudioConfig
from .data import list_pairs, list_pairs_by_split, load_segments, mel_spectrogram
from .infer import peak_pick
from .metrics import boundary_retrieval_fmeasure, framewise_accuracy
from .songformer import SongFormerConfig, SongFormerNet, save_songformer


IGNORE_INDEX = -100


def map_to_songformer_7(label: str) -> str:
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


def songformer_7_labels() -> List[str]:
    return ["silence", "intro", "verse", "chorus", "bridge", "outro", "inst"]


def gaussian_kernel(length: int, sigma: float) -> np.ndarray:
    length = int(length)
    if length <= 1:
        return np.array([1.0], dtype=np.float32)
    sigma = float(max(1e-6, sigma))
    x = np.arange(length, dtype=np.float32) - (length - 1) / 2.0
    w = np.exp(-0.5 * (x / sigma) ** 2)
    w = w / (w.sum() + 1e-8)
    return w.astype(np.float32)


def smooth_gaussian_1d(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    pad = int((len(kernel) - 1) // 2)
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="edge")
    y = np.convolve(xp, kernel, mode="valid")
    return y.astype(np.float32)


def downsample_max_ceil(x: np.ndarray, factor: int) -> np.ndarray:
    f = int(factor)
    if f <= 1:
        return x.astype(np.float32)
    t = int(x.shape[0])
    out_t = int(math.ceil(t / float(f)))
    pad = out_t * f - t
    if pad > 0:
        x = np.pad(x, (0, pad), mode="constant", constant_values=0.0)
    x = x.reshape(out_t, f)
    return x.max(axis=1).astype(np.float32)


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


def segments_to_frame_labels(
    segments, n_frames: int, audio_cfg: AudioConfig, label_to_id: Dict[str, int], duration_sec: float
) -> np.ndarray:
    y = np.full((n_frames,), int(IGNORE_INDEX), dtype=np.int64)
    frames_in_audio = int(min(n_frames, math.ceil(duration_sec * audio_cfg.sample_rate / audio_cfg.hop_length)))
    for seg in segments:
        lab = map_to_songformer_7(seg.label)
        lab_id = label_to_id.get(lab, int(IGNORE_INDEX))
        if lab_id == int(IGNORE_INDEX):
            continue
        s = int(max(0, math.floor(seg.start * audio_cfg.sample_rate / audio_cfg.hop_length)))
        e = int(max(s + 1, math.ceil(seg.end * audio_cfg.sample_rate / audio_cfg.hop_length)))
        s = min(s, frames_in_audio)
        e = min(e, frames_in_audio)
        if e > s:
            y[s:e] = int(lab_id)
    if frames_in_audio < n_frames:
        y[frames_in_audio:] = int(IGNORE_INDEX)
    return y


def segments_to_boundary_points(segments, n_frames: int, audio_cfg: AudioConfig, duration_sec: float) -> np.ndarray:
    y = np.zeros((n_frames,), dtype=np.float32)
    frames_in_audio = int(min(n_frames, math.ceil(duration_sec * audio_cfg.sample_rate / audio_cfg.hop_length)))
    for seg in segments[1:]:
        c = int(round(seg.start * audio_cfg.sample_rate / audio_cfg.hop_length))
        if 0 <= c < frames_in_audio:
            y[c] = 1.0
    return y


class SongFormerDataset(Dataset):
    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        audio_cfg: AudioConfig,
        max_seconds: float,
        downsample_factor: int,
    ):
        kept: List[Tuple[str, str]] = []
        for a, j in pairs:
            try:
                segs = load_segments(j)
            except Exception:
                continue
            if not segs:
                continue
            if float(segs[-1].end) <= 0:
                continue
            kept.append((a, j))
        self.pairs = kept
        self.audio_cfg = audio_cfg
        self.max_seconds = float(max_seconds)
        self.downsample_factor = int(downsample_factor)
        self.labels = songformer_7_labels()
        self.label_to_id = {l: i for i, l in enumerate(self.labels)}
        self.max_frames = int(math.ceil(self.max_seconds * audio_cfg.sample_rate / audio_cfg.hop_length))
        self.max_out_frames = int(math.ceil(self.max_frames / float(self.downsample_factor)))
        k = gaussian_kernel(length=11, sigma=11.0 / 6.0)
        self.boundary_smooth_kernel = k

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        audio_path, ann_path = self.pairs[idx]
        mel, _ = mel_spectrogram(audio_path, self.audio_cfg)
        mel = np.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
        segments = load_segments(ann_path)
        duration_sec = float(min(self.max_seconds, segments[-1].end if segments else self.max_seconds))
        mel = mel[:, : self.max_frames]
        if mel.shape[1] < self.max_frames:
            mel = np.pad(mel, ((0, 0), (0, self.max_frames - mel.shape[1])), mode="constant")
        fn_base = segments_to_frame_labels(segments, self.max_frames, self.audio_cfg, self.label_to_id, duration_sec)
        bd_base = segments_to_boundary_points(segments, self.max_frames, self.audio_cfg, duration_sec)
        fn_ds = downsample_mode_ceil(fn_base, self.downsample_factor, ignore_index=IGNORE_INDEX)
        bd_ds = downsample_max_ceil(bd_base, self.downsample_factor)
        bd_ds = smooth_gaussian_1d(bd_ds, self.boundary_smooth_kernel)
        if fn_ds.shape[0] < self.max_out_frames:
            fn_ds = np.pad(fn_ds, (0, self.max_out_frames - fn_ds.shape[0]), mode="constant", constant_values=int(IGNORE_INDEX))
        if bd_ds.shape[0] < self.max_out_frames:
            bd_ds = np.pad(bd_ds, (0, self.max_out_frames - bd_ds.shape[0]), mode="constant", constant_values=0.0)
        fn_ds = fn_ds[: self.max_out_frames]
        bd_ds = bd_ds[: self.max_out_frames]
        return {
            "mel": torch.from_numpy(mel.astype(np.float32)),
            "boundary_target": torch.from_numpy(bd_ds.astype(np.float32)),
            "function_target": torch.from_numpy(fn_ds.astype(np.int64)),
            "source_id": torch.tensor(0, dtype=torch.long),
            "duration_sec": torch.tensor(duration_sec, dtype=torch.float32),
        }


def softmax_focal_loss(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    logits = logits.float()
    b, t, c = logits.shape
    target = target.view(b, t)
    logp = torch.log_softmax(logits, dim=-1)
    p = logp.exp()
    valid = target != int(ignore_index)
    if not valid.any():
        return logits.new_tensor(0.0)
    idx = target.clamp(min=0).unsqueeze(-1)
    pt = p.gather(-1, idx).squeeze(-1)
    logpt = logp.gather(-1, idx).squeeze(-1)
    loss = -((1.0 - pt) ** float(gamma)) * logpt
    loss = loss[valid].mean()
    return loss


def boundary_aware_tv_loss(
    boundary_probs: torch.Tensor,
    boundary_target: torch.Tensor,
    beta: float = 0.6,
    alpha: float = 0.1,
    thr: float = 0.01,
) -> torch.Tensor:
    p = boundary_probs
    d = p[:, 1:] - p[:, :-1]
    w = torch.ones_like(d)
    m = (boundary_target > float(thr)).to(d.dtype)
    m_adj = torch.maximum(m[:, 1:], m[:, :-1])
    w = torch.where(m_adj > 0, torch.full_like(w, float(alpha)), w)
    loss = (w * d.abs().pow(float(beta))).mean()
    return loss


def evaluate_songformer(
    model: nn.Module,
    pairs: List[Tuple[str, str]],
    audio_cfg: AudioConfig,
    downsample_factor: int,
    device: torch.device,
    max_seconds: float,
) -> Dict[str, float]:
    model.eval()
    labels = songformer_7_labels()
    label_to_id = {l: i for i, l in enumerate(labels)}
    max_frames = int(math.ceil(float(max_seconds) * audio_cfg.sample_rate / audio_cfg.hop_length))
    frame_hop_sec = float(audio_cfg.hop_length * int(downsample_factor) / float(audio_cfg.sample_rate))
    hr5 = []
    hr3 = []
    accs = []
    with torch.no_grad():
        for audio_path, ann_path in tqdm(pairs, desc="evaluating", leave=False):
            segments = load_segments(ann_path)
            duration_sec = float(min(float(max_seconds), segments[-1].end if segments else float(max_seconds)))
            mel, _ = mel_spectrogram(audio_path, audio_cfg)
            mel = mel[:, :max_frames]
            if mel.shape[1] < max_frames:
                mel = np.pad(mel, ((0, 0), (0, max_frames - mel.shape[1])), mode="constant")
            mel_t = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0).to(device)
            out = model(mel_t, source_id=torch.zeros((1,), device=device, dtype=torch.long))
            boundary_logits = out["boundary_logits"].squeeze(0)
            function_logits = out["function_logits"].squeeze(0)
            bd_probs = torch.sigmoid(boundary_logits).detach().cpu().numpy()
            fn_probs = torch.softmax(function_logits, dim=-1).detach().cpu().numpy()
            picks = peak_pick(bd_probs, threshold=0.3, min_distance=64)
            pred_bounds = [float(p) * frame_hop_sec for p in picks if float(p) * frame_hop_sec < duration_sec]
            gt_bounds = [float(seg.start) for seg in segments[1:] if float(seg.start) < duration_sec]
            hr5.append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=0.5).get("f_measure", 0.0)))
            hr3.append(float(boundary_retrieval_fmeasure(pred_bounds, gt_bounds, tolerance=3.0).get("f_measure", 0.0)))
            t_out = fn_probs.shape[0]
            gt_frames = int(min(t_out, math.ceil(duration_sec / frame_hop_sec)))
            fn_gt_base = segments_to_frame_labels(segments, int(max_frames), audio_cfg, label_to_id, duration_sec)
            fn_gt = downsample_mode_ceil(fn_gt_base, int(downsample_factor), ignore_index=IGNORE_INDEX)
            fn_gt = fn_gt[:t_out]
            fn_gt[gt_frames:] = int(IGNORE_INDEX)
            times = [0.0] + sorted(pred_bounds) + [duration_sec]
            pred_ids = np.full((t_out,), int(IGNORE_INDEX), dtype=np.int64)
            for i in range(len(times) - 1):
                s = times[i]
                e = times[i + 1]
                fs = int(max(0, math.floor(s / frame_hop_sec)))
                fe = int(min(t_out, max(fs + 1, math.ceil(e / frame_hop_sec))))
                if fe <= fs:
                    continue
                avg = fn_probs[fs:fe].mean(axis=0)
                pred_lab = int(np.argmax(avg))
                pred_ids[fs:fe] = pred_lab
            pred_ids[gt_frames:] = int(IGNORE_INDEX)
            accs.append(framewise_accuracy(pred_ids, fn_gt, ignore_index=IGNORE_INDEX))
    return {
        "ACC": float(np.mean(accs)) if accs else 0.0,
        "HR.5F": float(np.mean(hr5)) if hr5 else 0.0,
        "HR3F": float(np.mean(hr3)) if hr3 else 0.0,
    }


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    step = int(step)
    warmup = int(max(1, warmup))
    total = int(max(warmup + 1, total))
    if step < warmup:
        return float(step) / float(warmup)
    progress = float(step - warmup) / float(total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def train_songformer(
    data_dir: str,
    out_path: str,
    epochs: int,
    batch_size: int,
    peak_lr: float,
    warmup_steps: int,
    total_steps: int,
    seed: int,
    max_seconds: float,
    downsample_factor: int,
    eval_interval: int,
    grad_accum: int = 1,
    amp: bool = True,
    log_interval: int = 0,
    audio_sample_rate: int = None,
    audio_n_mels: int = None,
    audio_hop_length: int = None,
    audio_n_fft: int = None,
):
    audio_cfg = AudioConfig(
        sample_rate=int(audio_sample_rate) if audio_sample_rate is not None else AudioConfig.sample_rate,
        n_mels=int(audio_n_mels) if audio_n_mels is not None else AudioConfig.n_mels,
        hop_length=int(audio_hop_length) if audio_hop_length is not None else AudioConfig.hop_length,
        n_fft=int(audio_n_fft) if audio_n_fft is not None else AudioConfig.n_fft,
    )
    rng = random.Random(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    splits = list_pairs_by_split(data_dir)
    if splits is None:
        all_pairs = list_pairs(data_dir)
        rng.shuffle(all_pairs)
        n = len(all_pairs)
        n_val = max(1, int(0.1 * n)) if n > 1 else 0
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]
    else:
        train_pairs, val_pairs, _ = splits
    if not train_pairs:
        raise ValueError("no train pairs found")

    labels = songformer_7_labels()
    s_cfg = SongFormerConfig(
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        downsample_factor=int(downsample_factor),
        global_pool=8,
        num_sources=1,
    )
    model = SongFormerNet(audio_cfg.n_mels, len(labels), cfg=s_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(peak_lr), weight_decay=0.01)
    if total_steps is None or int(total_steps) <= 0:
        steps_per_epoch = int(math.ceil(len(train_pairs) / float(batch_size)))
        total_steps = int(max(steps_per_epoch * int(epochs), 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=lambda s: cosine_with_warmup(s, warmup=int(warmup_steps), total=int(total_steps))
    )

    train_ds = SongFormerDataset(train_pairs, audio_cfg, max_seconds=max_seconds, downsample_factor=int(downsample_factor))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(batch_size),
        shuffle=True,
        drop_last=True,
        num_workers=max(1, min(os.cpu_count() or 1, 4)),
        pin_memory=pin,
    )

    metrics_path = out_path + ".metrics.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("")

    best_score = float("-inf")
    best_state = None
    bad = 0
    global_step = 0
    grad_accum = int(max(1, grad_accum))
    use_amp = bool(amp and device.type == "cuda")
    use_bf16 = bool(use_amp and torch.cuda.is_bf16_supported())
    scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp and (not use_bf16)))

    for epoch in range(int(epochs)):
        model.train()
        total_loss = 0.0
        total_n = 0
        skipped = 0
        opt.zero_grad(set_to_none=True)
        step_in_epoch = 0
        for batch in tqdm(train_loader, desc=f"songformer epoch {epoch+1}/{epochs}"):
            mel = batch["mel"].to(device, non_blocking=True)
            bd_t = batch["boundary_target"].to(device, non_blocking=True)
            fn_t = batch["function_target"].to(device, non_blocking=True)
            sid = batch["source_id"].to(device, non_blocking=True)
            mel = torch.nan_to_num(mel, nan=0.0, posinf=0.0, neginf=0.0)
            bd_t = torch.nan_to_num(bd_t, nan=0.0, posinf=0.0, neginf=0.0)
            autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                out = model(mel, source_id=sid)
                bd_logits = out["boundary_logits"]
                fn_logits = out["function_logits"]
            bd_logits_f = bd_logits.float()
            fn_logits_f = fn_logits.float()
            bd_t_f = bd_t.float()
            bce = F.binary_cross_entropy_with_logits(bd_logits_f, bd_t_f)
            bd_probs = torch.sigmoid(bd_logits_f)
            tv = boundary_aware_tv_loss(bd_probs, bd_t_f, beta=0.6, alpha=0.1, thr=0.01)
            ce_sum = F.cross_entropy(fn_logits_f.transpose(1, 2), fn_t, ignore_index=int(IGNORE_INDEX), reduction="sum")
            ce_den = torch.clamp((fn_t != int(IGNORE_INDEX)).sum().to(ce_sum.dtype), min=1.0)
            ce = ce_sum / ce_den
            fl = softmax_focal_loss(fn_logits_f, fn_t, gamma=2.0, ignore_index=int(IGNORE_INDEX))
            loss = 0.2 * (bce + 0.05 * tv) + 0.8 * (ce + 0.2 * fl)
            loss = loss / float(grad_accum)
            if not torch.isfinite(loss.detach()):
                opt.zero_grad(set_to_none=True)
                skipped += 1
                continue
            if int(log_interval) > 0 and (step_in_epoch % int(log_interval) == 0):
                tqdm.write(
                    f"step={step_in_epoch} loss={float(loss.detach().cpu().item()):.6f} "
                    f"bce={float(bce.detach().cpu().item()):.6f} tv={float(tv.detach().cpu().item()):.6f} "
                    f"ce={float(ce.detach().cpu().item()):.6f} fl={float(fl.detach().cpu().item()):.6f} "
                    f"skipped={skipped}"
                )
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.item()) * float(grad_accum)
            total_n += 1
            step_in_epoch += 1
            if step_in_epoch % grad_accum == 0:
                if scaler.is_enabled():
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                opt.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if global_step >= int(total_steps):
                    break
        if total_n == 0:
            raise SystemExit(f"all batches skipped due to non-finite loss (skipped={skipped})")
        train_loss = total_loss / float(total_n)

        do_eval = bool(val_pairs) and (
            int(eval_interval) <= 1 or ((epoch + 1) % int(eval_interval) == 0) or (epoch + 1) == int(epochs)
        )
        if do_eval:
            val = evaluate_songformer(
                model, val_pairs, audio_cfg, downsample_factor=int(downsample_factor), device=device, max_seconds=max_seconds
            )
            score = float(val["HR.5F"]) + float(val["ACC"])
            line = {"phase": "val", "epoch": epoch + 1, "train_loss": train_loss, "score": score, **val}
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
            if score > best_score:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 3:
                    break

        if global_step >= int(total_steps):
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    cfg = {
        "audio_cfg": asdict(audio_cfg),
        "songformer_cfg": asdict(s_cfg),
        "labels": labels,
        "max_seconds": float(max_seconds),
        "downsample_factor": int(downsample_factor),
        "best_score": float(best_score),
    }
    save_songformer(out_path, model, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_path", default="checkpoints/songformer.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--total_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seconds", type=float, default=420.0)
    parser.add_argument("--downsample_factor", type=int, default=3)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--no_amp", action="store_false", dest="amp")
    parser.set_defaults(amp=True)
    parser.add_argument("--log_interval", type=int, default=0)
    parser.add_argument("--audio_sample_rate", type=int, default=None)
    parser.add_argument("--audio_n_mels", type=int, default=None)
    parser.add_argument("--audio_hop_length", type=int, default=None)
    parser.add_argument("--audio_n_fft", type=int, default=None)
    args = parser.parse_args()
    train_songformer(
        args.data_dir,
        args.out_path,
        args.epochs,
        args.batch_size,
        args.peak_lr,
        args.warmup_steps,
        args.total_steps,
        args.seed,
        args.max_seconds,
        args.downsample_factor,
        args.eval_interval,
        args.grad_accum,
        args.amp,
        args.log_interval,
        args.audio_sample_rate,
        args.audio_n_mels,
        args.audio_hop_length,
        args.audio_n_fft,
    )


if __name__ == "__main__":
    main()
