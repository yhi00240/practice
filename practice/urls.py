from django.conf.urls import url

from practice import views

urlpatterns = [
    url(r'^main/', views.Main.as_view(), name='main'),

    url(r'^(?P<practice_name>.+)/data/$', views.Data.as_view(), name='data'),
    url(r'^(?P<practice_name>.+)/data/show_loaded_data/$', views.Data.show_loaded_data, name='data_show_loaded_data'),

    url(r'^(?P<practice_name>.+)/algorithm/$', views.Algorithm.as_view(), name='algorithm'),

    url(r'^(?P<practice_name>.+)/training/$', views.Training.as_view(), name='training'),
    url(r'^(?P<practice_name>.+)/training/check$', views.Training.check, name='training_check'),
    url(r'^(?P<practice_name>.+)/training/run$', views.Training.run, name='training_run'),
    url(r'^(?P<practice_name>.+)/training/run_service/$', views.Training.run_service, name='training_run_service'),
    url(r'^(?P<practice_name>.+)/training/get_progress/$', views.Training.get_progress, name='training_get_progress'),
    url(r'^(?P<practice_name>.+)/training/result$', views.Training.result, name='training_result'),

    url(r'^(?P<practice_name>.+)/test/$', views.Test.as_view(), name='test'),
    url(r'^(?P<practice_name>.+)/test/draw$', views.Test.draw, name='test_draw'),
    url(r'^(?P<practice_name>.+)/test/draw_result$', views.Test.draw_result, name='test_draw_result'),
]
