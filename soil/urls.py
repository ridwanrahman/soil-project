from django.urls import path

from . import views

urlpatterns = [
    # path('', views.train, name=''),
    path('train2', views.train_using_github_code, name='train_using_github_code'),
    # path('test', views.find_probability, name='find_probability'),
    path('index', views.index, name='index'),

    path('train-page', views.train_page, name='train'),
    path('test-page', views.test_page, name='test'),
    path('start-training', views.start_training, name='start_training'),
    path('get-test-result', views.testing, name='get-test-result')
]