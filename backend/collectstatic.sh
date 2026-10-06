#!/bin/bash
cd /app/
/opt/venv/bin/python manage.py collectstatic --noinput
