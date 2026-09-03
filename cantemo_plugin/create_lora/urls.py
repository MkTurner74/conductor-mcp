"""URLs for the CreateLora plugin."""
from django.urls import re_path

from . import views

urlpatterns = [
    # http://<portal_server_url>/create_lora/
    re_path(r"^$", views.CreateLoraFormView.as_view(), name="form"),
    # http://<portal_server_url>/create_lora/submit/
    re_path(r"^submit/$", views.CreateLoraSubmitView.as_view(), name="submit"),
]
