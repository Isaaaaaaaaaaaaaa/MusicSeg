set -e

ROOT=$(cd "$(dirname "$0")/.." && pwd)

if command -v redis-cli >/dev/null 2>&1; then
  if ! redis-cli ping >/dev/null 2>&1; then
    if command -v brew >/dev/null 2>&1; then
      brew services start redis >/dev/null 2>&1 || true
    fi
  fi
fi

if command -v redis-cli >/dev/null 2>&1; then
  if ! redis-cli ping >/dev/null 2>&1; then
    if command -v redis-server >/dev/null 2>&1; then
      redis-server --daemonize yes
    fi
  fi
fi

cd "$ROOT"
USE_CELERY=1 REDIS_URL="redis://localhost:6379/0" python3 backend/app.py > /tmp/musicseg-backend.log 2>&1 &

cd "$ROOT/backend"
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
PYTHONPATH="$ROOT/backend" python3 -m celery -A celery_worker.celery_app worker --loglevel=info --pool=solo --concurrency=1 > /tmp/musicseg-celery.log 2>&1 &

cd "$ROOT/frontend"
npm install > /tmp/musicseg-frontend-install.log 2>&1
npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/musicseg-frontend.log 2>&1 &

echo "MusicSeg started"
echo "Backend: http://127.0.0.1:5000"
echo "Frontend: http://127.0.0.1:5173"
