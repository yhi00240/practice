from django.db import models
# class InputData(models.Model):
#     name = models.CharField(max_length=10)
#     trainImages = models.BinaryField()
#     trainLabels = models.BinaryField()
#     testImages = models.BinaryField()
#     testLabels = models.BinaryField()


class TrainData(models.Model):
    image = models.BinaryField()
    label = models.IntegerField()

class TestData(models.Model):
    image = models.BinaryField()
    label = models.IntegerField()
