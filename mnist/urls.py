from django.conf.urls import url

from mnist import views

urlpatterns = [
    url(r'^', views.DataInput.as_view(), name='data_input'),
    url(r'^algorithm$', views.Algorithm.as_view(), name='algorithm'),
    url(r'^training$', views.Training.as_view(), name='training'),
    url(r'^test$', views.Test.as_view(), name='test'),
    url(r'^upload$', views.DataInput.as_view(), name='upload'),
]