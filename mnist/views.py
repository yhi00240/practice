from django.http import HttpResponse
from django.shortcuts import render
from rest_framework.renderers import TemplateHTMLRenderer
from rest_framework.views import APIView

class DataInput(APIView):
    renderer_classes = (TemplateHTMLRenderer,)
    template_name = 'mnist/data_input.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        return HttpResponse({'success': True})

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
