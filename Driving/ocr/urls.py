from django.conf.urls import  include, url
from ocr.views import *
urlpatterns = [
    url(r'^$', list, name='list'),
]

