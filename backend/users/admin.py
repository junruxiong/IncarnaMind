from django.contrib import admin

from .models import UserAccount

# Register your models here.


class UserAdmin(admin.ModelAdmin):
    list_display = (
        "email",
        "first_name",
        "last_name",
        "g_type",
        "subscription_end_date",
        "token_count",
        "max_tokens",
        "credits",
        "created_at",
        "updated_at",
        "last_login",
        "is_active",
    )


admin.site.register(UserAccount, UserAdmin)
