from django.urls import include, path
from rest_framework.routers import DefaultRouter

from .views import BlockViewSet, SessionViewSet

router = DefaultRouter()
router.register(r"mind", SessionViewSet, basename="mind")
# router.register(r"blocks", BlockViewSet, basename="block")
router.register(r"mind/(?P<mind_id>[^/.]+)/blocks", BlockViewSet, basename="block")


urlpatterns = [
    path("", include(router.urls)),
]
