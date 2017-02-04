import json
import os

from django.http import HttpResponse
from django.shortcuts import render, redirect
from rest_framework.decorators import detail_route
from rest_framework.viewsets import ViewSet

from practice.services import MNIST

PRACTICE_NAME_TO_CLASS = {
    'mnist': MNIST,
}

class PracticeViewSet(ViewSet):

    lookup_field = 'practice_name'

    @detail_route(methods=['get'])
    def input_data(self, request, practice_name=None):
        template_name = 'practice/data_input.html'
        return render(request, template_name, {'practice_name': practice_name})

    @detail_route(methods=['get'])
    def algorithm(self, request, practice_name=None):
        template_name = 'practice/set_algorithm.html'
        setting_list = PRACTICE_NAME_TO_CLASS[practice_name].get_algorithm_settings()
        return render(request, template_name, {'list': setting_list, 'practice_name': practice_name})

    @detail_route(methods=['get'])
    def training(self, request, practice_name=None):
        template_name = 'practice/set_training.html'
        setting_list = PRACTICE_NAME_TO_CLASS[practice_name].get_training_settings()
        return render(request, template_name, {'list': setting_list, 'practice_name': practice_name})

    @detail_route(methods=['get'])
    def run(self, request, practice_name=None):
        template_name = 'practice/run.html'
        return render(request, template_name, {'practice_name': practice_name, 'cookies_list': request.COOKIES})

    @detail_route(methods=['get'])
    def test(self, request, practice_name=None):
        template_name = 'practice/input_data.html'# Temporary
        return render(request, template_name, {'practice_name': practice_name})

    @detail_route(methods=['post'])
    def load_data(self, request, practice_name=None):
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        practice.load_training_data()
        # TODO : unable to save training data in session
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')

    @detail_route(methods=['post'])
    def upload(self, request, practice_name=None):
        up_file = request.FILES['file']
        if not os.path.exists('upload/'):
            os.mkdir('upload/')
        destination = open('upload/' + up_file.name, 'wb+')
        for chunk in up_file.chunks():
            destination.write(chunk)
        destination.close()
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')

    @detail_route(methods=['post'])
    def save_algorithm(self, request, practice_name=None):
        # 확인하면 주석 지워도 됩니당.
        # redirect하면서 저장된 쿠키값을 다시 사용자에게 보내주어야 하기 때문에 HttpResponse에 따로 저장한다.
        # 쿠키의 value는 string형으로 저장되므로 여기서는 int형이나 float형으로 변환하지 않는다.
        HttpResponse = redirect('/practice/' + practice_name + '/training/')
        HttpResponse.set_cookie('layers', request.data['Num of layers'])
        HttpResponse.set_cookie('activation_function', request.data['Activation Function'])
        HttpResponse.set_cookie('optimizer', request.data['Optimizer'])
        HttpResponse.set_cookie('weight_initialization', request.data['Weight Initialization'])
        HttpResponse.set_cookie('dropout', request.data['Dropout'])

        return HttpResponse

    @detail_route(methods=['post'])
    def save_training(self, request, practice_name=None):
        HttpResponse = redirect('/practice/' + practice_name + '/run/')
        HttpResponse.set_cookie('learning_rate', request.data['Learning Rate'])
        HttpResponse.set_cookie('optimization_epoch', request.data['Optimization Epoch'])

        return HttpResponse

    @detail_route(methods=['post'])
    def run_service(self, request, practice_name=None):
        mnist = MNIST()
        mnist.load_training_data()
        mnist.set_algorithm()
        mnist.set_training(float(request.COOKIES.get('learning_rate')), int(request.COOKIES.get('optimization_epoch')))
        message_list = mnist.run()
        return HttpResponse(json.dumps({'success': True, 'messages': message_list}), content_type='application/json')
