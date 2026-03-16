from datetime import datetime

from db import db


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(32), default="user", nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Token(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    token = db.Column(db.String(128), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class ModelVersion(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    boundary_ckpt = db.Column(db.String(255), nullable=False)
    classifier_ckpt = db.Column(db.String(255), nullable=False)
    is_active = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    duration = db.Column(db.Float)
    sample_rate = db.Column(db.Integer)
    segments = db.Column(db.JSON, nullable=False)
    waveform = db.Column(db.JSON, nullable=False)
    model_version_id = db.Column(db.Integer, db.ForeignKey("model_version.id"))
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class AnalysisJob(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    celery_id = db.Column(db.String(128))
    status = db.Column(db.String(32), default="pending", nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    analysis_id = db.Column(db.Integer, db.ForeignKey("analysis.id"))
    error = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)


class AnalysisLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action = db.Column(db.String(64), nullable=False)
    message = db.Column(db.Text)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    analysis_id = db.Column(db.Integer, db.ForeignKey("analysis.id"))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
