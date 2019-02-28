from django.conf.urls import  include, url
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.views.generic import RedirectView
from django.contrib import admin
# admin.autodiscover()
from ocr.views import index
urlpatterns = [
        url(r'^$',  include('ocr.urls')),
        url(r'^admin/', admin.site.urls),

] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


