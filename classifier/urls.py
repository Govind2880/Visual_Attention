from django.urls import path
from .views import home, predict_api, health_check, analytics_api, batch_predict_api

urlpatterns = [
    path("", home, name="home"),
    path("api/predict/", predict_api, name="predict_api"),
    path("api/batch-predict/", batch_predict_api, name="batch_predict_api"),
    path("api/health/", health_check, name="health_check"),
    path("api/analytics/", analytics_api, name="analytics_api"),
]
