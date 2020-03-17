import time
from importlib import import_module

from django.conf import settings
from main.models import ModelFile, WeightFile

class UploadedFileDeleteMiddleware(object):

    def __init__(self, get_response):
        self.get_response = get_response
    # One-time configuration and initialization.

    def __call__(self, request):
        # Code to be executed for each request before
        # the view (and later middleware) are called.
        response = self.get_response(request)
        empty = request.session.is_empty()
        # Code to be executed for each request/response after
        # the view is called.
        if request.method == 'POST':
            if settings.SESSION_COOKIE_NAME in request.COOKIES and empty:
                #ModelFile.session_expired = True
                #WeightFile.session_expired = True
                model_file = ModelFile.objects.filter(uploader=request.user)
                weight_file = WeightFile.objects.filter(uploader=request.user)
                model_file.session_expired = True
                weight_file.session_expired = True
                model_file.delete()
                weight_file.delete()

        return response