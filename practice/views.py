import os

from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.decorators import detail_route
from rest_framework.viewsets import ViewSet

from practice.services import MNIST

APP_NAME_TO_CLASS = {
    'mnist': MNIST,
}

class PracticeViewSet(ViewSet):

    lookup_field = 'app_name'

    @detail_route(methods=['get'])
    def input_data(self, request, app_name=None):
        template_name = 'practice/data_input.html'
        return render(self.request, template_name)

    @detail_route(methods=['post'])
    def upload(self, request, app_name=None):
        up_file = request.FILES['file']
        if not os.path.exists('upload/'):
            os.mkdir('upload/')
        destination = open('upload/' + up_file.name, 'wb+')
        for chunk in up_file.chunks():
            destination.write(chunk)
        destination.close()
        template_name = 'practice/data_input.html'
        return render(self.request, template_name)

    @detail_route(methods=['get'])
    def algorithm(self, request, app_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/set_algorithm.html'
        return render(self.request, template_name)

    @detail_route(methods=['get'])
    def training(self, request, app_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/set_training.html'
        return render(self.request, template_name)

    @detail_route(methods=['get'])
    def test(self, request, app_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/data_input.html'
        return render(self.request, template_name)
