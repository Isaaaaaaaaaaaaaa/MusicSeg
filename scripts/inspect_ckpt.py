import torch
import sys
import os

ckpt_path = "/Users/bytedance/MusicSeg/checkpoints/hx_boundary_v26/boundary.pt"
if not os.path.exists(ckpt_path):
    print(f"Checkpoint not found at {ckpt_path}")
    sys.exit(1)

try:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    print("Keys:", ckpt.keys())
    if "cfg" in ckpt:
        print("Config found.")
        cfg = ckpt["cfg"]
        print("Config keys:", cfg.keys())
        if "arch" in cfg:
            print("Arch:", cfg["arch"])
        if "boundary_cfg" in cfg:
            print("Boundary Config:", cfg["boundary_cfg"])
    elif "config" in ckpt:
        print("Config found as 'config'.")
        print(ckpt["config"])
    else:
        print("No config found.")
        
    if "state" in ckpt:
        print("State dict keys sample:", list(ckpt["state"].keys())[:5])
        # Check a specific weight shape
        for k in ckpt["state"]:
            if "linear1.weight" in k:
                print(f"{k}: {ckpt['state'][k].shape}")
                break
except Exception as e:
    print(f"Error loading checkpoint: {e}")
