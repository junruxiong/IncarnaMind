from celery import shared_task

from .models import UserAccount


@shared_task
def reset_user_credits():
    for user in UserAccount.objects.all():
        # if uers credts < than 20, set it to 20
        if user.credits < 20:
            user.credits = 20
            user.save()
