from django.db import models
from django.conf import settings
from django.contrib.auth.models import User
import os
import uuid
# Create your models here.

# if Model informations have changed, you should excute these commands;

# #1. python manage.py makemigrations
# #2. python manage.py migrate

class WeightFile(models.Model):
    def __str__(self):
        return self.filename()
    uploader = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    id = models.UUIDField(unique=True, primary_key=True, default=uuid.uuid4, help_text='Unique ID for weight file.')

    upload_date = models.DateField(null=True, auto_now_add=True)
    upload_time = models.TimeField(null=True, auto_now_add=True)
    weight_file = models.FileField(upload_to="weight_upload/")
    
    def filename(self):
        return os.path.basename(self.weight_file.name)

    def delete(self, *args, **kwargs):
        if os.path.isfile(self.weight_file.path):
            os.remove(self.weight_file.path)
        print(self.weight_file.path)
        super(WeightFile, self).delete(*args, **kwargs)

class ModelFile(models.Model):
    def __str__(self):
        return self.filename()
    
    uploader = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    id = models.UUIDField(unique=True, primary_key=True, default=uuid.uuid4, help_text='Unique ID for model file.')

    upload_date = models.DateField(null=True, auto_now_add=True)
    upload_time = models.TimeField(null=True, auto_now_add=True)
    model_file = models.FileField(upload_to="model_upload/")

    def filename(self):
        return os.path.basename(self.model_file.name)

    def delete(self, *args, **kwargs):
        if os.path.isfile(self.model_file.path):
            os.remove(self.model_file.path)
        print(self.model_file.path)
        super(ModelFile, self).delete(*args, **kwargs)