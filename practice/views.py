import json
import os

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from rest_framework.views import APIView

from practice.services import MNIST


def upload(request, practice_name=None):
    up_file = request.FILES['file']
    if not os.path.exists('upload/'):
        os.mkdir('upload/')
    destination = open('upload/' + up_file.name, 'wb+')
    for chunk in up_file.chunks():
        destination.write(chunk)
    destination.close()
    return HttpResponse(json.dumps({'success': True}), content_type='application/json')


class Data(APIView):

    template_name = 'practice/data.html'

    def get(self, request, practice_name):
        return render(request, self.template_name, {'practice_name': practice_name})

    def post(self, request, practice_name):
        # TODO : data 상태 화면에 보여주기.
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')


class Algorithm(APIView):

    template_name = 'practice/algorithm.html'

    def get(self, request, practice_name):
        setting_list = MNIST.get_algorithm_settings()
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        # redirect하면서 저장된 쿠키값을 다시 사용자에게 보내주어야 하기 때문에 HttpResponse에 따로 저장한다.
        # 쿠키의 value는 string형으로 저장되므로 여기서는 int형이나 float형으로 변환하지 않는다.
        HttpResponse = redirect(reverse('training', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('layers', request.data['Num of layers'])
        HttpResponse.set_cookie('activation_function', request.data['Activation Function'])
        HttpResponse.set_cookie('optimizer', request.data['Optimizer'])
        HttpResponse.set_cookie('weight_initialization', request.data['Weight Initialization'])
        HttpResponse.set_cookie('dropout', request.data['Dropout'])
        return HttpResponse


class Training(APIView):

    template_name = 'practice/training/training.html'

    def get(self, request, practice_name):
        setting_list = MNIST.get_training_settings()
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        HttpResponse = redirect(reverse('training_check', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('learning_rate', request.data['Learning Rate'])
        HttpResponse.set_cookie('optimization_epoch', request.data['Optimization Epoch'])
        return HttpResponse

    @staticmethod
    def check(request, practice_name):
        template = 'practice/training/check.html'
        return render(request, template, {'practice_name': practice_name, 'cookies_list': request.COOKIES})

    @staticmethod
    def run(request, practice_name):
        template = 'practice/training/run.html'
        return render(request, template, {'practice_name': practice_name})

    @staticmethod
    def run_service(request, practice_name):
        mnist = MNIST()
        mnist.load_training_data()
        mnist.set_algorithm()
        mnist.set_training(float(request.COOKIES.get('learning_rate')), int(request.COOKIES.get('optimization_epoch')))
        message_list = mnist.run()
        return HttpResponse(json.dumps({'success': True, 'messages': message_list}), content_type='application/json')

    @staticmethod
    def get_progress(request, practice_name):
        return HttpResponse(json.dumps({'success': True, 'messages': 'progress..'}), content_type='application/json')

    @staticmethod
    def result(request, practice_name):
        template = 'practice/training/run.html'
        return render(request, template, {'practice_name': practice_name, 'cookies_list': request.COOKIES})


class Test(APIView):

    template_name = 'practice/data.html' # TODO : implement test.html

    def get(self, request, practice_name):
        return render(request, self.template_name, {'practice_name': practice_name})
