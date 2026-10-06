from rest_framework import serializers

from .models import File


class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = File
        fields = [
            "id",
            "filename",
            "md5",
            "created_at",
            "updated_at",
        ]
