from django.conf import settings
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework_simplejwt.exceptions import AuthenticationFailed
from social_core.exceptions import AuthForbidden


class CustomJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        try:
            header = self.get_header(request)

            if header is None:
                raw_token = request.COOKIES.get(settings.AUTH_COOKIE)
            else:
                raw_token = self.get_raw_token(header)

            if raw_token is None:
                return None

            validated_token = self.get_validated_token(raw_token)
            user = self.get_user(validated_token)

            # Check if the user is active
            if not user.is_active:
                raise AuthenticationFailed("User is not active")

            return user, validated_token
        except AuthenticationFailed:
            # Handle specific authentication failures
            raise
        except:
            # Handle other exceptions
            return None


# def check_user_active(backend, user, response, *args, **kwargs):
#     if not user.is_active:
#         raise AuthForbidden(backend)
