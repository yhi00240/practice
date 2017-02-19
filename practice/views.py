import collections

import json

from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.views import APIView

from EasyTensor.redis_utils import RedisManager
from practice.services import MNIST
from practice.models import *

class Main(APIView):

    template_name = 'practice/main.html'

    def get(self,request):
        return render(request, self.template_name, {})

class Data(APIView):

    template_name = 'practice/data.html'

    def get(self, request, practice_name):
        return render(request, self.template_name, {'practice_name': practice_name})

    def post(self, request, practice_name):
        # TODO : data 상태 화면에 보여주기.
        return HttpResponse(json.dumps({'success': True}), content_type='application/json')

    @staticmethod
    @csrf_exempt
    def show_loaded_data(request, practice_name):
        images = []
        labels = []
        next = request.POST['next']
        for index in range(5*int(next)-4, 5*int(next)+1):
            data = TrainData.objects.get(id=index)
            images.append(list(data.image))
            labels.append(data.label)
        return HttpResponse(json.dumps({'success': True, 'images': images, 'labels': labels}), content_type='application/json')

class Algorithm(APIView):

    template_name = 'practice/algorithm.html'

    def get(self, request, practice_name):
        setting_list = collections.OrderedDict()
        setting_list['Model Type']={'Single layer', 'Multiple layers'}
        setting_list['Activation Function']={'Sigmoid', 'ReLU'}
        setting_list['Dropout']={'No', 'Yes'}
        setting_list['Weight Initialization']={'No', 'Yes'}
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        # redirect하면서 저장된 쿠키값을 다시 사용자에게 보내주어야 하기 때문에 HttpResponse에 따로 저장한다.
        # 쿠키의 value는 string형으로 저장되므로 여기서는 int형이나 float형으로 변환하지 않는다.
        HttpResponse = redirect(reverse('training', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('model_type', request.data.get('Model Type'))
        HttpResponse.set_cookie('activation_function', request.data.get('Activation Function'))
        HttpResponse.set_cookie('weight_initialization', request.data.get('Weight Initialization'))
        HttpResponse.set_cookie('dropout', request.data.get('Dropout'))
        return HttpResponse


class Training(APIView):

    template_name = 'practice/training/training.html'

    def get(self, request, practice_name):
        setting_list = collections.OrderedDict()
        setting_list['Optimizer']={'GradientDescentOptimizer', 'AdamOptimizer'}
        setting_list['Learning Rate']=0.01
        setting_list['Optimization Epoch']=10
        return render(request, self.template_name, {'list': setting_list, 'practice_name': practice_name})

    def post(self, request, practice_name):
        HttpResponse = redirect(reverse('training_check', kwargs={'practice_name':practice_name}))
        HttpResponse.set_cookie('optimizer', request.data.get('Optimizer'))
        HttpResponse.set_cookie('learning_rate', request.data.get('Learning Rate'))
        HttpResponse.set_cookie('optimization_epoch', request.data.get('Optimization Epoch'))
        return HttpResponse

    @staticmethod
    def check(request, practice_name):
        template = 'practice/training/check.html'
        algorithm_print_list = ['Model Type', 'Activation Function', 'Weight Initialization', 'Dropout']
        training_print_list = ['Optimizer', 'Learning Rate', 'Optimization Epoch']
        cookies = dict()
        cookies['algorithm'] = collections.OrderedDict()
        cookies['training'] = collections.OrderedDict()
        for item in algorithm_print_list:
            cookies['algorithm'][item] = request.COOKIES.get(item.replace(' ', '_').lower())
        for item in training_print_list:
            cookies['training'][item] = request.COOKIES.get(item.replace(' ', '_').lower())
        params = {
            'practice_name': practice_name,
            'algorithm_cookies': cookies['algorithm'],
            'training_cookies': cookies['training'],
        }
        return render(request, template, params)

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
        original, reference = mnist.test_single(image_data, request.COOKIES.get('model_type'), request.COOKIES.get('weight_initialization'))
        print('original', original)
        print('reference', reference)
        return HttpResponse(json.dumps({'original': original, 'reference': reference}), content_type='application/json')
