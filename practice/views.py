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
        return render(request, template_name, {'practice_name': practice_name})

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
        request.session['layers'] = request.data['Num of layers']
        request.session['activation_function'] = request.data['Activation Function']
        request.session['optimizer'] = request.data['Optimizer']
        request.session['weight_initialization'] = request.data['Weight Initialization']
        request.session['dropout'] = request.data['Dropout']
        return redirect('/practice/' + practice_name + '/training/')

    @detail_route(methods=['post'])
    def save_training(self, request, practice_name=None):
        request.session['learning_rate'] = float(request.data['Learning Rate'])
        request.session['optimization_epoch'] = int(request.data['Optimization Epoch'])
        return redirect('/practice/' + practice_name + '/run/')

    @detail_route(methods=['post'])
    def run_service(self, request, practice_name=None):
        mnist = MNIST()
        mnist.load_training_data()
        mnist.set_algorithm()
        mnist.set_training(request.session['learning_rate'], request.session['optimization_epoch'])
        message_list = mnist.run()
        return HttpResponse(json.dumps({'success': True, 'messages': message_list}), content_type='application/json')
