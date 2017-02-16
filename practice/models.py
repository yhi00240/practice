from django.db import models
class InputData(models.Model):
    name = models.CharField(max_length=10)
    trainImages = models.BinaryField()
    trainLabels = models.BinaryField()
    testImages = models.BinaryField()
    testLabels = models.BinaryField()