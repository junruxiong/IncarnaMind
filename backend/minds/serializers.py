from rest_framework import serializers

from .models import Block, Session


class SessionSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source="user.email")

    class Meta:
        model = Session
        fields = "__all__"


class BlockSerializer(serializers.ModelSerializer):
    user = serializers.ReadOnlyField(source="user.email")

    class Meta:
        model = Block
        fields = "__all__"
