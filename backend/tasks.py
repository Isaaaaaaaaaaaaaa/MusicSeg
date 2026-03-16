import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from celery import Celery
import librosa
import numpy as np
import redis as redis_lib

from db import db
from models import Analysis, AnalysisJob, AnalysisLog, ModelVersion
backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir.parents[0]))

from model.infer import analyze


def create_celery(app):
    broker = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    backend = os.getenv("CELERY_RESULT_BACKEND", broker)
    celery = Celery(app.import_name, broker=broker, backend=backend)
    celery.conf.update(app.config)
    if os.getenv("CELERY_EAGER") == "1":
        celery.conf.task_always_eager = True

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery


def create_analysis(audio_path: str, filename: str, model_version: ModelVersion, user_id: Optional[int]):
    result = analyze(audio_path, model_version.boundary_ckpt, model_version.classifier_ckpt)
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    waveform = y / (np.max(np.abs(y)) + 1e-6)
    max_points = 2000
    if len(waveform) > max_points:
        idx = np.linspace(0, len(waveform) - 1, max_points).astype(int)
        waveform = waveform[idx]
    analysis = Analysis(
        filename=filename,
        duration=result["duration"],
        sample_rate=sr,
        segments=result["segments"],
        waveform=waveform.tolist(),
        model_version_id=model_version.id,
        user_id=user_id,
    )
    db.session.add(analysis)
    db.session.commit()
    return analysis


def create_analysis_task(celery):
    @celery.task(name="analyze_audio")
    def analyze_audio(job_id: int, filename: str, tmp_path: str, model_version_id: int, user_id: Optional[int]):
        redis_url = os.getenv("REDIS_URL")
        redis_client = redis_lib.Redis.from_url(redis_url, decode_responses=True) if redis_url else None
        job = AnalysisJob.query.get(job_id)
        job.status = "running"
        db.session.commit()
        try:
            model_version = ModelVersion.query.get(model_version_id)
            analysis = create_analysis(tmp_path, filename, model_version, user_id)
            job.status = "done"
            job.analysis_id = analysis.id
            db.session.add(AnalysisLog(action="analysis_created", user_id=user_id, analysis_id=analysis.id))
            db.session.commit()
            if redis_client:
                payload = {
                    "id": uuid.uuid4().hex,
                    "title": "分析完成",
                    "message": filename,
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                }
                try:
                    redis_client.lpush("notifications", json.dumps(payload))
                    redis_client.ltrim("notifications", 0, 199)
                except Exception:
                    pass
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            db.session.commit()
            db.session.add(AnalysisLog(action="analysis_failed", user_id=user_id, message=str(e)))
            db.session.commit()
            if redis_client:
                payload = {
                    "id": uuid.uuid4().hex,
                    "title": "分析失败",
                    "message": filename,
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                }
                try:
                    redis_client.lpush("notifications", json.dumps(payload))
                    redis_client.ltrim("notifications", 0, 199)
                except Exception:
                    pass
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    return analyze_audio
