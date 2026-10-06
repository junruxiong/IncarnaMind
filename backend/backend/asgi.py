"""
ASGI config for backend project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/asgi/
"""

import os

from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application

from backend import routings

# Check for the WEBSITE_HOSTNAME environment variable to see if we are running in AKS
# If so, then load the settings from settings_prod.py
settings_module = (
    "backend.settings_prod" if "WEBSITE_HOSTNAME" in os.environ else "backend.settings"
)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)

application = ProtocolTypeRouter(
    {
        "http": get_asgi_application(),
        "websocket": URLRouter(routings.websocket_urlpatterns),
    }
)

# application = get_asgi_application()
