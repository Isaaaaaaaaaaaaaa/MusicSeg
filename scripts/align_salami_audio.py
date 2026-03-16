import argparse
import csv
import json
from pathlib import Path


def load_hx_segments(path: Path):
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except Exception:
        points = []
        segments = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                start, end, label = parts[0], parts[1], parts[2]
                segments.append({"start": float(start), "end": float(end), "label": str(label)})
            elif len(parts) == 2:
                t, label = parts[0], parts[1]
                points.append((float(t), str(label)))
        if segments:
            segments.sort(key=lambda x: (x["start"], x["end"]))
            return segments
        if len(points) >= 2:
            points.sort(key=lambda x: x[0])
            for i in range(len(points) - 1):
                start, label = points[i]
                end = points[i + 1][0]
                if str(label).lower() == "end":
                    continue
                segments.append({"start": float(start), "end": float(end), "label": str(label)})
            return segments
        return []

    if isinstance(data, dict):
        if "segments" in data and isinstance(data["segments"], list):
            items = data["segments"]
        elif "labels" in data and isinstance(data["labels"], list):
            items = data["labels"]
        else:
            items = []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    segments = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "start" in it and "end" in it:
            start = float(it["start"])
            end = float(it["end"])
            label = str(it.get("label", it.get("tag", it.get("name", "unknown"))))
            segments.append({"start": start, "end": end, "label": label})
            continue
        if "time" in it and "label" in it:
            start = float(it["time"])
            end = float(it.get("end", start))
            label = str(it["label"])
            segments.append({"start": start, "end": end, "label": label})
            continue
    segments.sort(key=lambda x: (x["start"], x["end"]))
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hx_jsonl", required=True, help="SongFormDB-HX jsonl 路径")
    parser.add_argument("--hx_root", required=True, help="SongFormDB 根目录（用于解析 label_path 相对路径）")
    parser.add_argument("--hx_audio_dir", default="", help="HX 重建后的音频目录（文件名使用 id.wav）")
    parser.add_argument("--hx_mel_dir", default="", help="HX mel npz 目录（默认 hx_root/HX）")
    parser.add_argument("--out_dir", required=True, help="输出目录")
    parser.add_argument("--training_manifest", default="", help="输出 training_manifest.csv 路径（可选）")
    parser.add_argument("--require_wav", action="store_true", default=False)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_audio = out_dir / "audio"
    out_ann = out_dir / "annotations"
    out_audio.mkdir(parents=True, exist_ok=True)
    out_ann.mkdir(parents=True, exist_ok=True)

    hx_root = Path(args.hx_root)
    hx_audio_dir = Path(args.hx_audio_dir) if args.hx_audio_dir else None
    hx_mel_dir = Path(args.hx_mel_dir) if args.hx_mel_dir else (hx_root / "HX")

    aligned_manifest = out_dir / "aligned_manifest.csv"
    training_manifest = Path(args.training_manifest) if args.training_manifest else (out_dir / "training_manifest.csv")

    ok_rows = []
    missing_wav = 0
    with aligned_manifest.open("w", newline="", encoding="utf-8") as mf:
        writer = csv.DictWriter(
            mf,
            fieldnames=["song_id", "audio_path", "annotation_path", "split", "status", "error"],
        )
        writer.writeheader()

        hx_path = Path(args.hx_jsonl)
        for line in hx_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            song_id = str(row.get("id") or row.get("song_id") or "").strip()
            if not song_id:
                continue
            split = str(row.get("split") or "train").strip().lower()
            label_path = row.get("label_path") or row.get("label") or row.get("annotation_path")
            if not label_path:
                writer.writerow({
                    "song_id": song_id,
                    "audio_path": "",
                    "annotation_path": "",
                    "split": split,
                    "status": "missing_label_path",
                    "error": "",
                })
                continue
            lp = Path(str(label_path))
            if not lp.is_absolute():
                lp = hx_root / lp
            try:
                segments = load_hx_segments(lp)
            except Exception as e:
                writer.writerow({
                    "song_id": song_id,
                    "audio_path": "",
                    "annotation_path": str(lp),
                    "split": split,
                    "status": "label_read_error",
                    "error": str(e),
                })
                continue
            ann_out = out_ann / f"{song_id}.json"
            ann_out.write_text(json.dumps({"segments": segments}, ensure_ascii=False, indent=2), encoding="utf-8")

            audio_path = ""
            if hx_audio_dir is not None:
                cand = hx_audio_dir / f"{song_id}.wav"
                if cand.exists():
                    out_path = out_audio / cand.name
                    if not out_path.exists():
                        out_path.write_bytes(cand.read_bytes())
                    audio_path = str(out_path)
            if not audio_path and not args.require_wav:
                mel_cand = hx_mel_dir / f"{song_id}.npz"
                if mel_cand.exists():
                    audio_path = str(mel_cand)
            if not audio_path and args.require_wav:
                missing_wav += 1
            status = "ok" if audio_path else "missing_audio"
            writer.writerow({
                "song_id": song_id,
                "audio_path": audio_path,
                "annotation_path": str(ann_out),
                "split": split,
                "status": status,
                "error": "",
            })
            if audio_path:
                ok_rows.append({
                    "song_id": song_id,
                    "audio_path": audio_path,
                    "annotation_path": str(ann_out),
                    "split": split,
                })

    training_manifest.parent.mkdir(parents=True, exist_ok=True)
    with training_manifest.open("w", newline="", encoding="utf-8") as tf:
        writer = csv.DictWriter(tf, fieldnames=["song_id", "audio_path", "annotation_path", "split"])
        writer.writeheader()
        for r in ok_rows:
            writer.writerow(r)
    if args.require_wav and missing_wav > 0:
        raise SystemExit(f"missing wav for {missing_wav} items")


if __name__ == "__main__":
    main()
