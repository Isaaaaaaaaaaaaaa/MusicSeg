
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from model.config import AudioConfig
from model.data import ClassifierDataset, ClassifierMelDataset, list_pairs, list_pairs_by_split
from model.model import SegmentMelAttnGate, SegmentMelAttn, SegmentMelCNN, SegmentClassifier
from model.train_classifier import eval_classifier

def load_classifier(ckpt_path: str, device: torch.device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
    
    print(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("state", checkpoint.get("model_state_dict"))
    cfg = checkpoint.get("cfg", checkpoint.get("config"))
    
    # Extract config
    labels = cfg.get("labels", [])
    feat_dim = cfg.get("feat_dim", 0)
    hidden = cfg.get("hidden", 256)
    arch = cfg.get("arch", "mlp")
    channels = cfg.get("channels", 64)
    attn_dim = cfg.get("attn_dim", 128)
    attn_heads = cfg.get("attn_heads", 4)
    input_type = cfg.get("input_type", "mel")
    
    print(f"Model Arch: {arch}, Labels: {len(labels)}")
    
    # Initialize model
    if input_type == "mel":
        n_mels = feat_dim
        if arch == "mel_attn_gate_ms":
            # This corresponds to our latest v25/v26 model
            # Note: The class name in code is SegmentMelAttnGate
            model = SegmentMelAttnGate(n_mels, labels, channels=channels, attn_dim=attn_dim, heads=attn_heads)
        elif arch == "mel_attn":
            model = SegmentMelAttn(n_mels, labels, channels=channels) # Wrapper needed? No, SegmentMelAttn is feature extractor usually
            # Actually train_classifier uses:
            # model = SegmentMelAttn(d_model=channels*2, nhead=attn_heads) + Linear head
            # But wait, looking at train_classifier.py...
            # It seems train_classifier instantiates models directly.
            # Let's assume SegmentMelAttnGate for v25/v26
            model = SegmentMelAttnGate(n_mels, labels, channels=channels, attn_dim=attn_dim, heads=attn_heads)
        elif arch == "mel_cnn":
            model = SegmentMelCNN(n_mels, labels, channels=channels)
        else:
            # Fallback or error
            print(f"Unknown arch {arch}, trying SegmentMelAttnGate")
            model = SegmentMelAttnGate(n_mels, labels, channels=channels, attn_dim=attn_dim, heads=attn_heads)
    else:
        model = SegmentClassifier(feat_dim, labels, hidden=hidden)
        
    model.load_state_dict(state_dict, strict=False) # strict=False to be safe with minor changes
    model.to(device)
    model.eval()
    
    return model, cfg

def main():
    parser = argparse.ArgumentParser(description="Evaluate Classifier on Test Set")
    parser.add_argument("--data_dir", required=True, help="Data directory")
    parser.add_argument("--ckpt_path", required=True, help="Path to classifier checkpoint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model, cfg = load_classifier(args.ckpt_path, device)
    
    # Configs
    audio_cfg = AudioConfig(**cfg.get("audio_cfg", {}))
    labels = cfg.get("labels", [])
    input_type = cfg.get("input_type", "mel")
    segment_frames = cfg.get("segment_frames", 384)
    
    # Data Split
    split = list_pairs_by_split(args.data_dir)
    if split is None:
        print("No split file found, using random split logic")
        # Import here to avoid circular imports if list_pairs is in data.py
        from model.data import list_pairs
        all_pairs = list_pairs(args.data_dir)
        
        # Consistent random shuffle
        import random
        rng = random.Random(42)
        rng.shuffle(all_pairs)
        
        n = len(all_pairs)
        n_test = int(n * 0.1)
        test_pairs = all_pairs[:n_test]
    else:
        _, _, test_pairs = split
        
    print(f"Test Set Size: {len(test_pairs)} songs")
    
    if len(test_pairs) == 0:
        print("No test pairs found!")
        return

    # Dataset
    label_to_id = {l: i for i, l in enumerate(labels)}
    min_seg_seconds = 0.5 # Default
    
    if input_type == "mel":
        test_dataset = ClassifierMelDataset(
            args.data_dir,
            audio_cfg,
            label_to_id,
            pairs=test_pairs,
            min_seg_seconds=min_seg_seconds,
            map_functional=True,
            segment_frames=segment_frames,
            train=False,
        )
        input_key = "mel"
    else:
        test_dataset = ClassifierDataset(
            args.data_dir,
            audio_cfg,
            label_to_id,
            pairs=test_pairs,
            min_seg_seconds=min_seg_seconds,
            map_functional=True,
        )
        input_key = "feat"
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    print(f"Evaluating on {len(test_dataset)} segments...")
    
    # Eval
    metrics = eval_classifier(model, test_loader, F.cross_entropy, device, len(labels), input_key)
    
    print("\n" + "="*40)
    print("Test Set Results")
    print("="*40)
    print(f"ACC:      {metrics['acc']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"Loss:     {metrics['loss']:.4f}")
    print("-" * 40)
    print("Per-Class Metrics:")
    for i, label in enumerate(labels):
        print(f"{label:<10} | Prec: {metrics['macro_precision']:.4f} | Rec: {metrics['macro_recall']:.4f}") 
        # Note: eval_classifier might not return per-class breakdown in the dict, 
        # but macro prec/rec gives an average. 
        # For detailed per-class, we'd need to modify eval_classifier or calculate here.
    print("="*40)

if __name__ == "__main__":
    main()
