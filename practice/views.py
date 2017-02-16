# -*- coding:utf-8 -*-
import json

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView

from EasyTensor.redis_utils import RedisManager
from practice.services import MNIST


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
        setting_list = {
            'Model Type': [
                'Single layer', 'Multiple layers'
            ],
            'Activation Function': [
                'Sigmoid', 'ReLU'
            ],
            'Optimizer': [
                'GradientDescentOptimizer', 'AdamOptimizer'
            ],
            'Weight Initialization': [
                'No', 'Yes'
            ],
            'Dropout': [
                'No', 'Yes'
            ]
        }
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        # redirect하면서 저장된 쿠키값을 다시 사용자에게 보내주어야 하기 때문에 HttpResponse에 따로 저장한다.
        # 쿠키의 value는 string형으로 저장되므로 여기서는 int형이나 float형으로 변환하지 않는다.
        HttpResponse = redirect(reverse('training', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('model_type', request.data.get('Model Type'))
        HttpResponse.set_cookie('activation_function', request.data.get('Activation Function'))
        HttpResponse.set_cookie('optimizer', request.data.get('Optimizer'))
        HttpResponse.set_cookie('weight_initialization', request.data.get('Weight Initialization'))
        HttpResponse.set_cookie('dropout', request.data.get('Dropout'))
        return HttpResponse


class Training(APIView):

    template_name = 'practice/training/training.html'

    def get(self, request, practice_name):
        setting_list = {
            'Learning Rate': 0.01,
            'Optimization Epoch': 10,
        }
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        HttpResponse = redirect(reverse('training_check', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('learning_rate', request.data.get('Learning Rate'))
        HttpResponse.set_cookie('optimization_epoch', request.data.get('Optimization Epoch'))
        return HttpResponse

    @staticmethod
    def check(request, practice_name):
        template = 'practice/training/check.html'
        print_list = ['Model type', 'Activation Function', 'Optimizer', 'Weight Initialization', 'Dropout', 'Learning Rate', 'Optimization Epoch']
        cookies_list = {}

        for item in print_list :
            cookies_list[item]=request.COOKIES.get(item.replace(' ','_').lower())

        return render(request, template, {'practice_name': practice_name, 'cookies_list': cookies_list})

    @staticmethod
    def run(request, practice_name):
        template = 'practice/training/run.html'
        return render(request, template, {'practice_name': practice_name})

    @staticmethod
    @csrf_exempt
    def run_service(request, practice_name):
        mnist = MNIST()
        mnist.load_data()
        mnist.set_algorithm(request.COOKIES.get('model_type'), request.COOKIES.get('weight_initialization'))
        mnist.set_training(request.COOKIES.get('optimizer'), float(request.COOKIES.get('learning_rate')), int(request.COOKIES.get('optimization_epoch')))
        RedisManager.delete(practice_name)
        mnist.run() # TODO : make async
        MNIST.tensorboard()
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')

    @staticmethod
    @csrf_exempt
    def get_progress(request, practice_name):
        message = RedisManager.get_message(practice_name)
        if not message:
            return HttpResponse(json.dumps({'success': False}), content_type='application/json')
        else:
            return HttpResponse(json.dumps({'success': True, 'messages': str(message, 'utf-8')}), content_type='application/json')

    @staticmethod
    def result(request, practice_name):
        template = 'practice/training/result.html'
        return render(request, template, {'practice_name': practice_name})


class Test(APIView):

    template_name = 'practice/test/test.html'

    def get(self, request, practice_name):
        return render(request, self.template_name, {'practice_name': practice_name})

    @staticmethod
    def draw(request, practice_name):
        template = 'practice/test/draw.html'
        return render(request, template, {'practice_name': practice_name, 'keys': range(MNIST.NUM_CLASSES)})

    @staticmethod
    @csrf_exempt
    def draw_result(request, practice_name):
        image_data = eval(request.POST['image_data'])
        mnist = MNIST()
        results = mnist.test_single(image_data, request.COOKIES.get('model_type'), request.COOKIES.get('weight_initialization'))
        # TODO : select model type to test
        print(results)
        return HttpResponse(json.dumps({'results': results}), content_type='application/json')
