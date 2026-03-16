import os
import sys
import glob
import json
import logging
import uvicorn
import torch
import numpy as np
import librosa
import shutil
import time
import traceback
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# Ensure model can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model inference utilities
from model.infer import (
    load_boundary, 
    load_classifier, 
    compute_mel, 
    sliding_boundary_probs, 
    songformer_pick_boundaries,
    refine_peaks
)
from model.config import AudioConfig, BoundaryConfig

# Configuration
AUDIO_DIR = "/Users/bytedance/MusicSeg/data/songform-hx-aligned/audio/"
UPLOAD_DIR = "/Users/bytedance/MusicSeg/data/uploads/"
CHECKPOINT_DIR = "/Users/bytedance/MusicSeg/checkpoints/hx_boundary_v26/"
BOUNDARY_CKPT = os.path.join(CHECKPOINT_DIR, "boundary.pt")
CLASSIFIER_CKPT = os.path.join(CHECKPOINT_DIR, "classifier.pt")
HOST = "0.0.0.0"
PORT = 8000

# Create upload dir
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MusicSeg API", description="SOTA Music Structure Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
models = {}

@app.on_event("startup")
async def startup_event():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading models on {device}...")
    
    try:
        if os.path.exists(BOUNDARY_CKPT):
            try:
                models["boundary"] = load_boundary(BOUNDARY_CKPT)
                models["boundary"]["model"].to(device)
                models["boundary"]["model"].eval()
                logger.info("Boundary model loaded.")
            except Exception as e:
                logger.error(f"Failed to load boundary model: {e}")
                traceback.print_exc()
        else:
            logger.warning(f"Boundary checkpoint not found at {BOUNDARY_CKPT}")

        if os.path.exists(CLASSIFIER_CKPT):
            try:
                models["classifier"] = load_classifier(CLASSIFIER_CKPT)
                models["classifier"]["model"].to(device)
                models["classifier"]["model"].eval()
                logger.info("Classifier model loaded.")
            except Exception as e:
                logger.error(f"Failed to load classifier model: {e}")
                traceback.print_exc()
        else:
            logger.warning(f"Classifier checkpoint not found at {CLASSIFIER_CKPT}")
            
        models["device"] = device
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        traceback.print_exc()

@app.get("/api/health")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

@app.get("/api/songs")
def list_songs():
    """List all audio files in the directory."""
    songs = []
    
    # Scan standard audio dir
    if os.path.exists(AUDIO_DIR):
        files = glob.glob(os.path.join(AUDIO_DIR, "*.wav")) + \
                glob.glob(os.path.join(AUDIO_DIR, "*.mp3")) + \
                glob.glob(os.path.join(AUDIO_DIR, "*.flac"))
        for f in sorted(files):
            filename = os.path.basename(f)
            songs.append({
                "filename": filename,
                "path": f,
                "name": os.path.splitext(filename)[0].replace("HX_", "").replace("_", " ").title(),
                "source": "database"
            })
            
    # Scan upload dir
    if os.path.exists(UPLOAD_DIR):
        files = glob.glob(os.path.join(UPLOAD_DIR, "*.wav")) + \
                glob.glob(os.path.join(UPLOAD_DIR, "*.mp3")) + \
                glob.glob(os.path.join(UPLOAD_DIR, "*.flac"))
        for f in sorted(files):
            filename = os.path.basename(f)
            songs.append({
                "filename": filename,
                "path": f,
                "name": os.path.splitext(filename)[0].replace("_", " ").title() + " (Uploaded)",
                "source": "upload"
            })
            
    return {"songs": songs}

@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Upload an audio file."""
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "filename": file.filename,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/audio/{filename}")
def get_audio(filename: str):
    """Stream audio file."""
    # Check both dirs
    file_path_db = os.path.join(AUDIO_DIR, filename)
    file_path_up = os.path.join(UPLOAD_DIR, filename)
    
    if os.path.exists(file_path_db):
        return FileResponse(file_path_db, media_type="audio/wav")
    elif os.path.exists(file_path_up):
        return FileResponse(file_path_up, media_type="audio/wav")
    else:
        raise HTTPException(status_code=404, detail="File not found")

class SegmentResult(BaseModel):
    start: float
    end: float
    label: str
    confidence: Optional[float] = 1.0

class AnalysisResult(BaseModel):
    filename: str
    duration: float
    segments: List[SegmentResult]
    inference_time: float

@app.post("/api/predict/{filename}", response_model=AnalysisResult)
async def predict(filename: str):
    """Run SOTA Music Structure Analysis on the file."""
    start_time = time.time()
    
    # Check boundary model first
    if "boundary" not in models or models["boundary"] is None:
        logger.error("Boundary model is missing or failed to load")
        raise HTTPException(status_code=503, detail="Boundary model not loaded")
    
    # Locate file
    file_path_db = os.path.join(AUDIO_DIR, filename)
    file_path_up = os.path.join(UPLOAD_DIR, filename)
    
    if os.path.exists(file_path_db):
        file_path = file_path_db
    elif os.path.exists(file_path_up):
        file_path = file_path_up
    else:
        raise HTTPException(status_code=404, detail="File not found")

    device = models["device"]
    
    # 1. Load Audio & Compute Mel
    logger.info(f"Processing {filename}...")
    try:
        # Load audio using librosa
        # Using sr=None to keep original sampling rate initially, but we might need resampling
        y, sr = librosa.load(file_path, sr=None) 
        duration = float(librosa.get_duration(y=y, sr=sr))
        
        audio_cfg = models["boundary"]["audio_cfg"]
        if isinstance(audio_cfg, dict):
            audio_cfg = AudioConfig(**audio_cfg)
            
        # Resample for model if needed
        if sr != audio_cfg.sample_rate:
            y = librosa.resample(y, orig_sr=sr, target_sr=audio_cfg.sample_rate)
            
        mel = compute_mel(y, audio_cfg) # (n_mels, T)
        
    except Exception as e:
        logger.error(f"Audio processing failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Audio processing failed: {str(e)}")

    # 2. Boundary Detection
    try:
        boundary_model = models["boundary"]["model"]
        boundary_cfg = models["boundary"]["boundary_cfg"]
        if isinstance(boundary_cfg, dict):
            boundary_cfg = BoundaryConfig(**boundary_cfg)
            
        # Run inference
        probs = sliding_boundary_probs(boundary_model, mel, boundary_cfg)
        
        # Pick peaks
        picks = songformer_pick_boundaries(probs, audio_cfg, boundary_cfg)
        
        # Convert frames to seconds
        frame_time = audio_cfg.hop_length / audio_cfg.sample_rate
        boundaries_sec = [float(p * frame_time) for p in picks]
        
        # Add start (0.0) and end (duration) if missing
        boundaries_sec = sorted(list(set([0.0] + boundaries_sec + [duration])))
        
    except Exception as e:
        logger.error(f"Boundary detection failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Boundary detection failed: {str(e)}")

    # 3. Segment Classification
    segments = []
    # Force unknown if classifier is missing OR if we caught an error earlier
    classifier_available = ("classifier" in models and models["classifier"] is not None)
    
    if classifier_available:
        try:
            classifier_model = models["classifier"]["model"]
            classifier_cfg = models["classifier"] # dict
            labels = classifier_cfg["labels"]
            segment_frames = classifier_cfg.get("segment_frames", 256)
            
            # Prepare batch of segments
            segment_mels = []
            valid_indices = []
            
            for i in range(len(boundaries_sec) - 1):
                start_sec = boundaries_sec[i]
                end_sec = boundaries_sec[i+1]
                
                if end_sec - start_sec < 0.5: # Skip very short segments
                    continue
                    
                start_frame = int(start_sec / frame_time)
                end_frame = int(end_sec / frame_time)
                
                # Extract segment mel
                if end_frame > mel.shape[1]:
                    end_frame = mel.shape[1]
                
                if start_frame >= end_frame:
                    continue

                seg_mel = mel[:, start_frame:end_frame]
                
                # Pad/Crop to fixed size for classifier
                if seg_mel.shape[1] < segment_frames:
                    pad = segment_frames - seg_mel.shape[1]
                    seg_mel = np.pad(seg_mel, ((0,0), (0, pad)), mode='constant')
                else:
                    # Center crop
                    center = seg_mel.shape[1] // 2
                    start = max(0, center - segment_frames // 2)
                    seg_mel = seg_mel[:, start:start+segment_frames]
                    
                segment_mels.append(seg_mel)
                valid_indices.append(i)
                
            if segment_mels:
                batch_tensor = torch.from_numpy(np.stack(segment_mels)).to(device)
                
                with torch.no_grad():
                    logits = classifier_model(batch_tensor)
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    confs = torch.softmax(logits, dim=-1).max(dim=-1).values.cpu().numpy()
                
                # Construct result
                for idx, pred_idx, conf in zip(valid_indices, preds, confs):
                    segments.append(SegmentResult(
                        start=boundaries_sec[idx],
                        end=boundaries_sec[idx+1],
                        label=labels[pred_idx],
                        confidence=float(conf)
                    ))
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            traceback.print_exc()
            # If classification fails mid-way, clear segments to trigger fallback
            segments = []
    
    # Fallback: Fill all segments with "unknown" if list is empty
    # This happens if classifier is missing OR if classification crashed
    if not segments and len(boundaries_sec) > 1:
         for i in range(len(boundaries_sec) - 1):
             if boundaries_sec[i+1] - boundaries_sec[i] >= 0.5:
                 segments.append(SegmentResult(start=boundaries_sec[i], end=boundaries_sec[i+1], label="unknown", confidence=0.0))
    
    # Sort segments by start time
    segments.sort(key=lambda x: x.start)
    
    end_time = time.time()
    inference_time = end_time - start_time

    return AnalysisResult(
        filename=filename,
        duration=duration,
        segments=segments,
        inference_time=inference_time
    )

# Mount Frontend (Production Build)
# Check for frontend build relative to this file
frontend_dist = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../frontend/dist")
if os.path.exists(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
else:
    @app.get("/")
    async def read_index():
         return {"message": "MusicSeg API Running. Please start frontend with 'cd frontend && npm run dev'"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
