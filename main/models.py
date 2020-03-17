from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_delete
from django.dispatch import receiver
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
        self.weight_file.delete()
        super().delete(*args, **kwargs)
        
@receiver(post_delete, sender=WeightFile)
def weight_submission_delete(sender, instance, **kwargs):
    instance.weight_file.delete(False) 


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
        self.model_file.delete()
        super().delete(*args, **kwargs)

@receiver(post_delete, sender=ModelFile)
def model_submission_delete(sender, instance, **kwargs):
    instance.model_file.delete(False) 