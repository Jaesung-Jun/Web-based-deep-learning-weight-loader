
from django import forms
from .models import WeightFile, ModelFile

class WeightFileUploadForm(forms.ModelForm):
    class Meta:
        model = WeightFile
        fields = ['weight_file']

class ModelFileUploadForm(forms.ModelForm):
    class Meta:
        model = ModelFile
        fields = ['model_file']