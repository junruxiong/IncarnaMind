"""
https://docs.celeryq.dev/en/stable/django/first-steps-with-django.html
"""

import os

from celery import Celery
from celery.schedules import crontab, timedelta
from django.conf import settings

# this code copied from manage.py
# set the default Django settings module for the 'celery' app.
# When running on AKS you should use the settings_prod settings.
settings_module = (
    "backend.settings_prod" if "WEBSITE_HOSTNAME" in os.environ else "backend.settings"
)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

# you can change the name here
app = Celery("backend")

# read config from Django settings, the CELERY namespace would make celery
# config keys has `CELERY` prefix
app.config_from_object("django.conf:settings", namespace="CELERY")

# discover and load tasks.py from from all registered Django apps
app.autodiscover_tasks(lambda: settings.INSTALLED_APPS)

app.conf.beat_schedule = {
    "reset-credits": {
        "task": "users.tasks.reset_user_credits",
        # Executes every 20 hours
        "schedule": timedelta(hours=20),
    },
}
