import json
import os
import secrets
import tempfile
import sys
import uuid
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS
import librosa
import numpy as np
import redis as redis_lib
from werkzeug.security import check_password_hash, generate_password_hash

from db import db
from models import Analysis, AnalysisJob, AnalysisLog, ModelVersion, Token, User
from tasks import create_analysis, create_analysis_task, create_celery

backend_dir = Path(__file__).resolve().parent
sys.path.append(str(backend_dir))
sys.path.append(str(backend_dir.parents[1]))



def create_app():
    app = Flask(__name__)
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        db_path = Path(__file__).resolve().parent / "app.db"
        database_url = f"sqlite:///{db_path}"
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    CORS(app)
    db.init_app(app)

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis_lib.Redis.from_url(redis_url, decode_responses=True) if redis_url else None

    celery = create_celery(app)
    analyze_task = create_analysis_task(celery)
    async_state = {"enabled": os.getenv("USE_CELERY") == "1"}

    with app.app_context():
        db.create_all()
        ensure_default_admin()
        ensure_default_model()

    def require_auth(admin_only=False):
        def decorator(fn):
            @wraps(fn)
            def wrapper(*args, **kwargs):
                token_value = request.headers.get("Authorization", "").replace("Bearer ", "")
                if not token_value:
                    return jsonify({"error": "unauthorized"}), 401
                token = Token.query.filter_by(token=token_value).first()
                if not token or token.expires_at < datetime.utcnow():
                    return jsonify({"error": "unauthorized"}), 401
                user = db.session.get(User, token.user_id)
                if not user:
                    return jsonify({"error": "unauthorized"}), 401
                if admin_only and user.role != "admin":
                    return jsonify({"error": "forbidden"}), 403
                request.current_user = user
                return fn(*args, **kwargs)

            return wrapper

        return decorator

    def log_action(action, message="", analysis_id=None, user_id=None):
        entry = AnalysisLog(action=action, message=message, analysis_id=analysis_id, user_id=user_id)
        db.session.add(entry)
        db.session.commit()
        push_notification(action, message, user_id)

    def resolve_ckpt_path(value, fallback):
        if value:
            p = Path(value)
            if p.exists():
                return str(p)
            candidate = backend_dir.parent / value
            if candidate.exists():
                return str(candidate)
        for item in fallback:
            if item.exists():
                return str(item)
        return str(fallback[0])

    def push_notification(action, message, user_id):
        if not redis_client:
            return
        title_map = {
            "analysis_created": "分析完成",
            "analysis_queued": "分析排队",
            "analysis_deleted": "分析删除",
            "analysis_updated": "标注更新",
            "model_activated": "模型切换",
            "user_created": "新增用户",
            "login": "用户登录",
            "logout": "用户退出",
        }
        payload = {
            "id": uuid.uuid4().hex,
            "title": title_map.get(action, action),
            "message": message or action,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
        }
        try:
            redis_client.lpush("notifications", json.dumps(payload))
            redis_client.ltrim("notifications", 0, 199)
        except Exception:
            return

    @app.route("/api/health", methods=["GET"])
    def health():
        redis_reachable = False
        if redis_client:
            try:
                redis_client.ping()
                redis_reachable = True
            except Exception:
                redis_reachable = False
        return jsonify({
            "status": "ok",
            "async_enabled": async_state["enabled"],
            "redis_enabled": bool(redis_client),
            "redis_reachable": redis_reachable,
            "queue_enabled": async_state["enabled"] and redis_reachable,
            "redis_url": redis_url,
        })

    @app.route("/api/async/enable", methods=["POST"])
    @require_auth(admin_only=True)
    def enable_async():
        if not redis_client:
            return jsonify({"error": "redis not configured"}), 400
        try:
            redis_client.ping()
        except Exception:
            return jsonify({"error": "redis not reachable"}), 400
        async_state["enabled"] = True
        return jsonify({"async_enabled": True, "queue_enabled": True})

    @app.route("/api/auth/login", methods=["POST"])
    def login():
        data = request.get_json(silent=True) or {}
        username = data.get("username", "")
        password = data.get("password", "")
        user = User.query.filter_by(username=username).first()
        if not user or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "invalid credentials"}), 401
        token_value = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=12)
        token = Token(token=token_value, user_id=user.id, expires_at=expires_at)
        db.session.add(token)
        db.session.commit()
        log_action("login", user_id=user.id)
        return jsonify({"token": token_value, "role": user.role, "username": user.username})

    @app.route("/api/auth/logout", methods=["POST"])
    @require_auth()
    def logout():
        token_value = request.headers.get("Authorization", "").replace("Bearer ", "")
        token = Token.query.filter_by(token=token_value).first()
        if token:
            db.session.delete(token)
            db.session.commit()
        log_action("logout", user_id=request.current_user.id)
        return jsonify({"status": "ok"})

    @app.route("/api/auth/me", methods=["GET"])
    @require_auth()
    def me():
        user = request.current_user
        return jsonify({"id": user.id, "username": user.username, "role": user.role})

    @app.route("/api/users", methods=["GET"])
    @require_auth(admin_only=True)
    def list_users():
        users = User.query.order_by(User.created_at.desc()).all()
        return jsonify([
            {"id": u.id, "username": u.username, "role": u.role, "created_at": u.created_at.isoformat()}
            for u in users
        ])

    @app.route("/api/users", methods=["POST"])
    @require_auth(admin_only=True)
    def create_user():
        data = request.get_json(silent=True) or {}
        username = data.get("username")
        password = data.get("password")
        role = data.get("role", "user")
        if not username or not password:
            return jsonify({"error": "username and password required"}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "username exists"}), 400
        user = User(username=username, password_hash=generate_password_hash(password, method="pbkdf2:sha256"), role=role)
        db.session.add(user)
        db.session.commit()
        log_action("user_created", user_id=request.current_user.id, message=username)
        return jsonify({"id": user.id})

    @app.route("/api/users/<int:user_id>", methods=["PUT"])
    @require_auth(admin_only=True)
    def update_user(user_id: int):
        data = request.get_json(silent=True) or {}
        user = User.query.get_or_404(user_id)
        if "role" in data:
            user.role = data["role"]
        if "password" in data and data["password"]:
            user.password_hash = generate_password_hash(data["password"], method="pbkdf2:sha256")
        db.session.commit()
        log_action("user_updated", user_id=request.current_user.id, message=user.username)
        return jsonify({"status": "ok"})

    @app.route("/api/users/<int:user_id>", methods=["DELETE"])
    @require_auth(admin_only=True)
    def delete_user(user_id: int):
        user = User.query.get_or_404(user_id)
        db.session.delete(user)
        db.session.commit()
        log_action("user_deleted", user_id=request.current_user.id, message=user.username)
        return jsonify({"status": "deleted"})

    @app.route("/api/models", methods=["GET"])
    @require_auth(admin_only=True)
    def list_models():
        models = ModelVersion.query.order_by(ModelVersion.created_at.desc()).all()
        return jsonify([
            {
                "id": m.id,
                "name": m.name,
                "boundary_ckpt": m.boundary_ckpt,
                "classifier_ckpt": m.classifier_ckpt,
                "is_active": m.is_active,
                "created_at": m.created_at.isoformat(),
            }
            for m in models
        ])

    @app.route("/api/models", methods=["POST"])
    @require_auth(admin_only=True)
    def create_model():
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        boundary_ckpt = data.get("boundary_ckpt")
        classifier_ckpt = data.get("classifier_ckpt")
        if not name or not boundary_ckpt or not classifier_ckpt:
            return jsonify({"error": "invalid payload"}), 400
        model = ModelVersion(name=name, boundary_ckpt=boundary_ckpt, classifier_ckpt=classifier_ckpt)
        db.session.add(model)
        db.session.commit()
        log_action("model_created", user_id=request.current_user.id, message=name)
        return jsonify({"id": model.id})

    @app.route("/api/models/<int:model_id>/activate", methods=["POST"])
    @require_auth(admin_only=True)
    def activate_model(model_id: int):
        model = ModelVersion.query.get_or_404(model_id)
        ModelVersion.query.update({ModelVersion.is_active: False})
        model.is_active = True
        db.session.commit()
        log_action("model_activated", user_id=request.current_user.id, message=model.name)
        return jsonify({"status": "ok"})

    @app.route("/api/analyze", methods=["POST"])
    @require_auth()
    def api_analyze():
        if "file" not in request.files:
            return jsonify({"error": "file required"}), 400
        file = request.files["file"]
        model_version = ModelVersion.query.filter_by(is_active=True).first()
        if not model_version:
            return jsonify({"error": "model not configured"}), 500
        if not Path(model_version.boundary_ckpt).exists() or not Path(model_version.classifier_ckpt).exists():
            model_version.boundary_ckpt = resolve_ckpt_path(
                model_version.boundary_ckpt,
                [backend_dir.parent / "checkpoints" / "best_boundary.pt", backend_dir.parent / "checkpoints" / "boundary.pt"],
            )
            model_version.classifier_ckpt = resolve_ckpt_path(
                model_version.classifier_ckpt,
                [backend_dir.parent / "checkpoints" / "best_classifier.pt", backend_dir.parent / "checkpoints" / "classifier.pt"],
            )
            db.session.commit()
        use_async = request.args.get("async") == "1"
        if use_async:
            if not async_state["enabled"]:
                return jsonify({"error": "async not enabled"}), 400
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                file.save(tmp.name)
                audio_path = tmp.name
            job = AnalysisJob(filename=file.filename, user_id=request.current_user.id)
            db.session.add(job)
            db.session.commit()
            try:
                task = analyze_task.delay(job.id, file.filename, audio_path, model_version.id, request.current_user.id)
                job.celery_id = task.id
                db.session.commit()
                log_action("analysis_queued", user_id=request.current_user.id, message=file.filename)
                return jsonify({"job_id": job.id, "status": job.status})
            except Exception as e:
                job.status = "failed"
                job.error = str(e)
                db.session.commit()
                log_action("analysis_failed", user_id=request.current_user.id, message=str(e))
                if os.path.exists(audio_path):
                    os.remove(audio_path)
                return jsonify({"error": str(e)}), 500

        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            file.save(tmp.name)
            audio_path = tmp.name
        try:
            analysis = create_analysis(audio_path, file.filename, model_version, request.current_user.id)
            log_action("analysis_created", user_id=request.current_user.id, analysis_id=analysis.id)
            return jsonify({
                "id": analysis.id,
                "segments": analysis.segments,
                "duration": analysis.duration,
                "waveform": analysis.waveform,
                "sample_rate": analysis.sample_rate,
            })
        except Exception as e:
            log_action("analysis_failed", user_id=request.current_user.id, message=str(e))
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)

    @app.route("/api/jobs", methods=["GET"])
    @require_auth()
    def list_jobs():
        page, page_size = parse_pagination()
        query = AnalysisJob.query.order_by(AnalysisJob.created_at.desc())
        total = query.count()
        items = query.offset((page - 1) * page_size).limit(page_size).all()
        return jsonify({
            "items": [
                {
                    "id": j.id,
                    "filename": j.filename,
                    "status": j.status,
                    "error": j.error,
                    "analysis_id": j.analysis_id,
                    "created_at": j.created_at.isoformat(),
                }
                for j in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        })

    @app.route("/api/jobs/<int:job_id>", methods=["GET"])
    @require_auth()
    def get_job(job_id: int):
        job = AnalysisJob.query.get_or_404(job_id)
        return jsonify({
            "id": job.id,
            "filename": job.filename,
            "status": job.status,
            "error": job.error,
            "analysis_id": job.analysis_id,
            "created_at": job.created_at.isoformat(),
        })

    @app.route("/api/analyses", methods=["GET"])
    @require_auth()
    def list_analyses():
        page, page_size = parse_pagination()
        keyword = request.args.get("query", "").strip()
        query = Analysis.query
        if keyword:
            query = query.filter(Analysis.filename.ilike(f"%{keyword}%"))
        total = query.count()
        items = query.order_by(Analysis.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
        return jsonify({
            "items": [
                {
                    "id": item.id,
                    "filename": item.filename,
                    "duration": item.duration,
                    "created_at": item.created_at.isoformat(),
                }
                for item in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        })

    @app.route("/api/analyses/<int:analysis_id>", methods=["GET"])
    @require_auth()
    def get_analysis(analysis_id: int):
        item = Analysis.query.get_or_404(analysis_id)
        return jsonify({
            "id": item.id,
            "filename": item.filename,
            "duration": item.duration,
            "sample_rate": item.sample_rate,
            "segments": item.segments,
            "waveform": item.waveform,
            "created_at": item.created_at.isoformat(),
        })

    @app.route("/api/analyses/<int:analysis_id>", methods=["PUT"])
    @require_auth()
    def update_analysis(analysis_id: int):
        data = request.get_json(silent=True) or {}
        segments = data.get("segments")
        if segments is None or not isinstance(segments, list):
            return jsonify({"error": "segments required"}), 400
        item = Analysis.query.get_or_404(analysis_id)
        item.segments = segments
        db.session.commit()
        log_action("analysis_updated", user_id=request.current_user.id, analysis_id=analysis_id)
        return jsonify({
            "id": item.id,
            "filename": item.filename,
            "duration": item.duration,
            "sample_rate": item.sample_rate,
            "segments": item.segments,
            "waveform": item.waveform,
            "created_at": item.created_at.isoformat(),
        })

    @app.route("/api/analyses/<int:analysis_id>", methods=["DELETE"])
    @require_auth(admin_only=True)
    def delete_analysis(analysis_id: int):
        item = Analysis.query.get_or_404(analysis_id)
        db.session.delete(item)
        db.session.commit()
        log_action("analysis_deleted", user_id=request.current_user.id, analysis_id=analysis_id)
        return jsonify({"status": "deleted"})

    @app.route("/api/logs", methods=["GET"])
    @require_auth(admin_only=True)
    def list_logs():
        page, page_size = parse_pagination()
        total = AnalysisLog.query.count()
        items = AnalysisLog.query.order_by(AnalysisLog.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
        return jsonify({
            "items": [
                {
                    "id": l.id,
                    "action": l.action,
                    "message": l.message,
                    "user_id": l.user_id,
                    "analysis_id": l.analysis_id,
                    "created_at": l.created_at.isoformat(),
                }
                for l in items
            ],
            "total": total,
            "page": page,
            "page_size": page_size,
        })

    @app.route("/api/notifications", methods=["GET"])
    @require_auth()
    def list_notifications():
        page, page_size = parse_pagination()
        if not redis_client:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})
        try:
            total = redis_client.llen("notifications")
            start = (page - 1) * page_size
            end = start + page_size - 1
            raw_items = redis_client.lrange("notifications", start, end)
            items = [json.loads(item) for item in raw_items]
            return jsonify({"items": items, "total": total, "page": page, "page_size": page_size})
        except Exception:
            return jsonify({"items": [], "total": 0, "page": page, "page_size": page_size})

    @app.route("/api/analyses/summary", methods=["GET"])
    @require_auth()
    def analyses_summary():
        items = Analysis.query.all()
        total_duration = 0.0
        total_segments = 0
        label_counts = {}
        for item in items:
            if item.duration:
                total_duration += item.duration
            for seg in item.segments or []:
                label = seg.get("label", "unknown")
                label_counts[label] = label_counts.get(label, 0) + 1
                total_segments += 1
        avg_duration = round(total_duration / len(items), 2) if items else 0
        return jsonify({
            "avg_duration": avg_duration,
            "total_segments": total_segments,
            "label_counts": label_counts,
        })

    @app.route("/api/stats", methods=["GET"])
    @require_auth()
    def get_stats():
        total = Analysis.query.count()
        return jsonify({"total_analyses": total})

    return app


def parse_pagination():
    try:
        page = int(request.args.get("page", 1))
    except ValueError:
        page = 1
    try:
        page_size = int(request.args.get("page_size", 20))
    except ValueError:
        page_size = 20
    page = max(page, 1)
    page_size = min(max(page_size, 1), 100)
    return page, page_size


def ensure_default_admin():
    if User.query.count() > 0:
        return
    username = os.getenv("DEFAULT_ADMIN_USER", "admin")
    password = os.getenv("DEFAULT_ADMIN_PASS", "admin123")
    user = User(username=username, password_hash=generate_password_hash(password, method="pbkdf2:sha256"), role="admin")
    db.session.add(user)
    db.session.commit()


def ensure_default_model():
    if ModelVersion.query.count() > 0:
        return
    project_root = backend_dir.parent
    boundary_env = os.getenv("BOUNDARY_CKPT")
    classifier_env = os.getenv("CLASSIFIER_CKPT")

    def resolve_path(value, fallback):
        if value:
            p = Path(value)
            if p.exists():
                return str(p)
            candidate = project_root / value
            if candidate.exists():
                return str(candidate)
        for item in fallback:
            if item.exists():
                return str(item)
        return str(fallback[0])

    boundary_path = resolve_path(
        boundary_env,
        [project_root / "checkpoints" / "best_boundary.pt", project_root / "checkpoints" / "boundary.pt"],
    )
    classifier_path = resolve_path(
        classifier_env,
        [project_root / "checkpoints" / "best_classifier.pt", project_root / "checkpoints" / "classifier.pt"],
    )
    model = ModelVersion(name="default", boundary_ckpt=boundary_path, classifier_ckpt=classifier_path, is_active=True)
    db.session.add(model)
    db.session.commit()


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
