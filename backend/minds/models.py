import uuid

from django.db import models

from users.models import UserAccount

# Create your models here.


class Session(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(UserAccount, on_delete=models.CASCADE)
    name = models.CharField(max_length=255, blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name}"

    # class Meta:
    #     verbose_name = "Session"
    #     verbose_name_plural = "Sessions"


class Block(models.Model):
    class MindType(models.TextChoices):
        TEXT = "text"
        QUERY = "query"
        OUTPUT = "output"

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=True)
    client_id = models.CharField(max_length=255, blank=True, null=True)
    order = models.IntegerField(blank=True, null=True)
    user = models.ForeignKey(UserAccount, on_delete=models.CASCADE)
    mind_id = models.ForeignKey(Session, on_delete=models.CASCADE)

    is_prompt = models.BooleanField(default=True)
    type = models.CharField(
        max_length=10, choices=MindType.choices, default=MindType.QUERY
    )
    model = models.CharField(max_length=255, blank=True, null=True)

    text = models.TextField(blank=True, null=True)
    metadata = models.JSONField(blank=True, null=True)
    # metadata = models.TextField(blank=True, null=True)
    # ! metadata: retrieval citation, sequance number and others
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    reply_to = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="reply"
    )
    query_to = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="query"
    )

    def __str__(self):
        return f"{self.id}"
