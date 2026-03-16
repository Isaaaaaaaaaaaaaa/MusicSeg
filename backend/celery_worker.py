from app import create_app
from tasks import create_analysis_task, create_celery

flask_app = create_app()
celery_app = create_celery(flask_app)
create_analysis_task(celery_app)
