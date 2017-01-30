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
        return render(request, template_name)

    @detail_route(methods=['post'])
    def load_data(self, request, practice_name=None):
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        practice.load_training_data()
        return HttpResponse({'data': practice.training_data})

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
        return render(self.request, template_name)

    @detail_route(methods=['get'])
    def algorithm(self, request, practice_name=None):
        template_name = 'practice/set_algorithm.html'
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        return render(request, template_name, practice.get_algorithm_settings())

    @detail_route(methods=['post'])
    def save_algorithm(self, request, app_name=None):
        self.request.session['Layers'] = self.request.data['Layers']
        self.request.session['Activationfunction'] = self.request.data['Activationfunction']
        self.request.session['Optimizer'] = self.request.data['Optimizer']
        self.request.session['WeightInitialization'] = self.request.data['WeightInitialization']
        self.request.session['Dropout'] = self.request.data['Dropout']
        return redirect('/practice/mnist/training/')

    @detail_route(methods=['get'])
    def training(self, request, practice_name=None):
        template_name = 'practice/set_training.html'
        practice = PRACTICE_NAME_TO_CLASS[practice_name]()
        return render(request, template_name, practice.get_training_settings())

    @detail_route(methods=['post'])
    def save_training(self, request, app_name=None):
        self.request.session['rate'] = int(self.request.data['rate'])
        self.request.session['epoch'] = int(self.request.data['epoch'])
        return redirect('/practice/mnist/training_mnist/')

    @detail_route(methods=['get'])
    def training_mnist(self, request, app_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/training_mnist.html'
        mnist = MNIST()
        mnist.loadTrainingData()
        mnist.setAlgorithm()
        mnist.setTraining(self.request.session['epoch'])
        mnist.run()

        return render(self.request, template_name, {'user_epoch': mnist.training_epochs})

    @detail_route(methods=['get'])
    def test(self, request, practice_name=None):
        # TODO for front: 알맞은 html template 개발되면 적용
        template_name = 'practice/data_input.html'
        return render(self.request, template_name)
