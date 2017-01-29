from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.views import APIView
import os

class DataInput(APIView):
    renderer_classes = (TemplateHTMLRenderer,)
    template_name = 'mnist/data_input.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request, format='None'):
        up_file = request.FILES['file']
        if not os.path.exists('upload/'):
            os.mkdir('upload/')
        destination = open('upload/'+up_file.name, 'wb+')
        for chunk in up_file.chunks():
            destination.write(chunk)
        destination.close()
        return render(request, self.template_name)

class Algorithm(APIView):
    renderer_classes = (TemplateHTMLRenderer,)
    template_name = 'mnist/data_input.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        return HttpResponse({'success': True})

class Training(APIView):
    renderer_classes = (TemplateHTMLRenderer,)
    template_name = 'mnist/data_input.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        return HttpResponse({'success': True})

class Test(APIView):
    renderer_classes = (TemplateHTMLRenderer,)
    template_name = 'mnist/data_input.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        return HttpResponse({'success': True})
