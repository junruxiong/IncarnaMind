import os

from .settings import *  # noqa
from .settings import BASE_DIR

print("Prduction settings")


ALLOWED_HOSTS = (
    [
        # os.environ["CUSTOM_HOSTNAME"],
        "api.incarnamind.com",
    ]
    if "WEBSITE_HOSTNAME" in os.environ
    else []
)


CSRF_TRUSTED_ORIGINS = (
    [
        # "https://" + os.environ["CUSTOM_HOSTNAME"],
        "https://api.incarnamind.com",
    ]
    if "WEBSITE_HOSTNAME" in os.environ
    else []
)

CORS_ALLOWED_ORIGINS = (
    [
        # os.environ["WEBSITE_HOSTNAME"],
        # f"https://{os.environ['CUSTOM_HOSTNAME']}",
        # f"https://{os.environ['FRONTEND_HOSTNAME']}",
        "https://incarnamind.com",
        "https://www.incarnamind.com",
        "https://api.incarnamind.com",
    ]
    if "WEBSITE_HOSTNAME" in os.environ
    else []
)


CORS_ALLOW_CREDENTIALS = True
DEBUG = False


# # ! WhiteNoise configuration
# MIDDLEWARE = [
#     "django.middleware.security.SecurityMiddleware",
#     # Add whitenoise middleware after the security middleware
#     "whitenoise.middleware.WhiteNoiseMiddleware",
#     "django.contrib.sessions.middleware.SessionMiddleware",
#     "corsheaders.middleware.CorsMiddleware",
#     "django.middleware.common.CommonMiddleware",
#     "django.middleware.csrf.CsrfViewMiddleware",
#     "django.contrib.auth.middleware.AuthenticationMiddleware",
#     "django.contrib.messages.middleware.MessageMiddleware",
#     "django.middleware.clickjacking.XFrameOptionsMiddleware",
# ]


# STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"
# STATIC_ROOT = os.path.join(BASE_DIR, "staticfiles")


# STATICFILES_STORAGE = "backend.azure_storage.AzureStaticStorage"
# DEFAULT_FILE_STORAGE = "backend.azure_storage.AzureMediaStorage"

STORAGES = {
    "default": {
        "BACKEND": "backend.azure_storage.AzureMediaStorage",
        # Additional configuration for media storage
    },
    "staticfiles": {
        "BACKEND": "backend.azure_storage.AzureStaticStorage",
        # Additional configuration for static files storage
    },
}

AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_ACCOUNT_KEY = os.getenv("AZURE_ACCOUNT_KEY")
AZURE_CUSTOM_DOMAIN = f"{AZURE_ACCOUNT_NAME}.blob.core.windows.net"

STATIC_URL = f"https://{AZURE_CUSTOM_DOMAIN}/static/"
STATIC_ROOT = os.path.join(BASE_DIR, "static")

MEDIA_URL = f"https://{AZURE_CUSTOM_DOMAIN}/media/"
MEDIA_ROOT = os.path.join(BASE_DIR, "media")


DB_IGNORE_SSL = os.environ.get("DB_IGNORE_SSL") == "true"
DATABASES = {
    "default": {
        "ENGINE": os.environ.get("DATABASE_ENGINE", "django.db.backends.sqlite3"),
        "HOST": os.environ.get("DATABASE_HOST", "localhost"),
        "NAME": os.environ.get("DATABASE_NAME", os.path.join(BASE_DIR, "db.sqlite3")),
        "USER": os.environ.get("DATABASE_USER", "user"),
        "PASSWORD": os.environ.get("DATABASE_PASSWORD", "password"),
        "PORT": os.environ.get("DATABASE_PORT", "5432"),
    }
}
if not DB_IGNORE_SSL:
    DATABASES["default"]["OPTIONS"] = {"sslmode": "require"}
