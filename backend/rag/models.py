import os
import uuid
from typing import Any

from django.db import models

from users.models import UserAccount

# Create your models here.


def upload_to(instance, filename):
    # upload to media url
    return f"files/{instance.user.id}/{filename}"


class File(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    filename = models.CharField(max_length=256, blank=False, null=False)
    # file_name = models.CharField(max_length=256, blank=False, null=False)
    type = models.CharField(max_length=128, blank=False, null=False)
    user = models.ForeignKey(UserAccount, on_delete=models.CASCADE)
    dir = models.FileField(upload_to=upload_to, blank=False, null=False)
    md5 = models.CharField(max_length=32, blank=False, null=False)
    token_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def delete(self, *args: Any, **kwargs: Any) -> None:
        self.dir.delete(save=False)
        super().delete(*args, **kwargs)

    def __str__(self):
        return f"{self.dir.name}"

    # def file_name(self):
    #     return os.path.basename(self.dir.name)
