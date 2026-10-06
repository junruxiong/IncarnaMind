from django.contrib import admin

from .models import File

# Register your models here.


class RagAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "filename",
        "token_count",
        "type",
        "user",
        "dir",
        "md5",
        "created_at",
        "updated_at",
    )
    ordering = ("created_at",)


admin.site.register(File, RagAdmin)
