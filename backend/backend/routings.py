from django.urls import path, re_path

from minds import consumers

# websocket_urlpatterns = [
#     # xxxxxxx/room/x1
#     # ws://127.0.0.1:8000/room/群号/
#     re_path(r"room/(?P<group>\w+)/$", consumers.IncarnaConsumer.as_asgi()),
# ]
websocket_urlpatterns = [
    path("<str:mind_id>/<str:userId>", consumers.IncarnaConsumer.as_asgi())
]
