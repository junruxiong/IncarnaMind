FROM --platform=linux/amd64 python:3.11

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV GOOGLE_APPLICATION_CREDENTIALS /var/secrets/google/keyfile.json

# Make a new directory to put our app in.
RUN mkdir /app
# RUN mkdir /app/staticfiles
# RUN mkdir /app/mediafiles
# Copy to app folder
COPY . /app/
# Change the working directory.
WORKDIR /app

RUN apt-get update \
    # dependencies for building Python packages
    && apt-get install -y build-essential \
    # psycopg2 dependencies
    && apt-get install -y libpq-dev \
    # Translations dependencies
    && apt-get install -y gettext \
    # libGL dependency for OpenCV
    && apt-get install -y libgl1 \
    # Additional dependencies
    && apt-get install -y procps telnet \
    # cleaning up unused files
    && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && rm -rf /var/lib/apt/lists/*

# Requirements are installed here to ensure they will be cached.

COPY ./requirements.txt /requirements.txt
RUN pip install 'setuptools<72.0.0'
RUN pip install -r /requirements.txt

# CMD ["gunicorn", "backend.wsgi:application", "--bind", "0.0.0.0:8000", "--log-level", "info"]
CMD python manage.py runserver 0.0.0.0:8000