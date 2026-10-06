from asgiref.sync import async_to_sync
from channels.exceptions import StopConsumer
from channels.generic.websocket import JsonWebsocketConsumer, WebsocketConsumer
from django.contrib.auth import get_user_model


class IncarnaConsumer(JsonWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mind_id = None
        self.user = None

    def connect(self):
        # from users.models import UserAccount
        # from .models import Block, Session

        User = get_user_model()

        self.accept()
        self.mind_id = self.scope["url_route"]["kwargs"]["mind_id"]
        self.user = User.objects.get(id=self.scope["url_route"]["kwargs"]["userId"])
        # 将这个客户端的连接对象加入到某个地方（内存 or redis）
        async_to_sync(self.channel_layer.group_add)(self.mind_id, self.channel_name)
        print("mind_id: ", self.mind_id)

    def receive_json(self, content):
        mind_id = self.mind_id
        user = self.user
        message = content["message"]
        print("user: ", content)

        async_to_sync(self.channel_layer.group_send)(
            self.mind_id,
            {
                "type": "chat.message",
                "new_message": content["message"],
            },
        )

    def chat_message(self, event):
        self.send_json(event)

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(self.mind_id, self.channel_name)
        super().disconnect(close_code)

    # def connect(self, message):
    #     # 接收这个客户端的连接
    #     self.accept()

    #     # 获取群号，获取路由匹配中的
    #     group = self.scope["url_route"]["kwargs"].get("group")

    #     # 将这个客户端的连接对象加入到某个地方（内存 or redis）
    #     async_to_sync(self.channel_layer.group_add)(group, self.channel_name)

    # def receive(self, message):
    #     group = self.scope["url_route"]["kwargs"].get("group")

    #     # 通知组内的所有客户端，执行 xx_oo 方法，在此方法中自己可以去定义任意的功能。
    #     async_to_sync(self.channel_layer.group_send)(
    #         group, {"type": "xx.oo", "message": message}
    #     )

    # def xx_oo(self, event):
    #     text = event["message"]["text"]
    #     self.send(text)

    # def disconnect(self, message):
    #     group = self.scope["url_route"]["kwargs"].get("group")

    #     async_to_sync(self.channel_layer.group_discard)(group, self.channel_name)
    #     raise StopConsumer()
