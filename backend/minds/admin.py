from django.contrib import admin

from .models import Block, Session

# Register your models here.


class SessionAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "name",
        "created_at",
        "updated_at",
    )
    ordering = ("created_at",)
    search_fields = (
        "user__email",
        "name",
    )
    list_filter = (
        "user__email",
        "name",
    )


class BlockAdmin(admin.ModelAdmin):
    list_display = (
        "user",
        "mind_id",
        "type",
        "order",
        "text",
        "created_at",
        "updated_at",
        "query_to",
        "reply_to",
        "id",
    )
    ordering = ("mind_id", "order")
    search_fields = (
        "user__email",
        "mind_id",
        "type",
        "text",
    )
    list_filter = (
        "user__email",
        "mind_id",
        "type",
    )


admin.site.register(Session, SessionAdmin)
admin.site.register(Block, BlockAdmin)
