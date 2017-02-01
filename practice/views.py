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
        return render(request, template_name, {'practice_name':practice_name})

    @detail_route(methods=['post'])
    def load_data(self, request, practice_name=None):
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        practice.load_training_data()
        return HttpResponse({'success': True, 'data': practice.training_data, 'practice_name':practice_name}, content_type='application/json')

    @detail_route(methods=['post'])
    def upload(self, request, practice_name=None):
        up_file = request.FILES['file']
        if not os.path.exists('upload/'):
            os.mkdir('upload/')
        destination = open('upload/' + up_file.name, 'wb+')
        for chunk in up_file.chunks():
            destination.write(chunk)
        destination.close()
        template_name = 'practice/data_input.html'
        return render(request, template_name, {'practice_name':practice_name})

    @detail_route(methods=['get'])
    def algorithm(self, request, practice_name=None):
        template_name = 'practice/set_algorithm.html'
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        return render(request, template_name, {'list': practice.get_algorithm_settings(),'practice_name':practice_name})

    @detail_route(methods=['post'])
    def save_algorithm(self, request, practice_name=None):
        request.session['Layers'] = request.data['Layers']
        request.session['Activationfunction'] = request.data['Activationfunction']
        request.session['Optimizer'] = request.data['Optimizer']
        request.session['WeightInitialization'] = request.data['WeightInitialization']
        request.session['Dropout'] = request.data['Dropout']
        return redirect('/practice/' + practice_name + 'training/')

    @detail_route(methods=['get'])
    def training(self, request, practice_name=None):
        template_name = 'practice/set_training.html'
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        return render(request, template_name, {'list': practice.get_training_settings(), 'practice_name':practice_name})

    @detail_route(methods=['post'])
    def save_training(self, request, practice_name=None):
        request.session['rate'] = float(request.data['rate'])
        request.session['epoch'] = int(request.data['epoch'])
        return redirect('/practice/' + practice_name + '/training_mnist/')

    @detail_route(methods=['get'])
    def training_mnist(self, request, practice_name=None):
        template_name = 'practice/training_mnist.html'
        mnist = MNIST()
        mnist.load_training_data()
        mnist.set_algorithm()
        mnist.set_training(request.session['rate'], request.session['epoch'])
        mnist.run()
        return render(request, template_name, {'user_epoch': mnist.training_epochs, 'practice_name':practice_name})

    @detail_route(methods=['get'])
    def test(self, request, practice_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/data_input.html'
        return render(request, template_name, {'practice_name':practice_name})
