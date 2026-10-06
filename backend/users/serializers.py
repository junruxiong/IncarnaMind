from djoser.serializers import UserSerializer as BaseUserSerializer
from rest_framework import serializers

from .models import UserAccount  # Import your custom user model


class CustomUserSerializer(BaseUserSerializer):
    g_type = serializers.IntegerField()
    token_count = serializers.IntegerField()
    max_tokens = serializers.IntegerField()
    credits = serializers.IntegerField()
    current_model = serializers.CharField()
    is_retrieval = serializers.BooleanField()

    class Meta(BaseUserSerializer.Meta):
        model = UserAccount
        fields = BaseUserSerializer.Meta.fields + (
            "g_type",
            "token_count",
            "max_tokens",
            "credits",
            "current_model",
            "is_retrieval",
        )
