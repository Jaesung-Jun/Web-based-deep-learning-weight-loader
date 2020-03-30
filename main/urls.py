"""weight_loader URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))

★ Caution!
    + In urlpattern dictionary, [path] function's first arguments are CANNOT same.
    + ex)   path('<pk>', views.weightDetailView, name='weight-detail'),
    +       path('<pk>', views.modelDetailView, name='model-detail'),
    + this example is error.

"""
from django.urls import path
from django.conf.urls import url
from . import views



urlpatterns = [
    path('', views.uploadFile, name='upload'),
    path('weight/<pk>', views.weightDetailView, name='weight-detail'),
    path('model/<pk>', views.modelDetailView, name='model-detail'),
    path('success/<model_pk>/<weight_pk>', views.uploadSuccess, name='upload-success'),
    #url(r'^success/(?P<uploader>\w+)/$', views.upload_success, name='upload-success'),
]