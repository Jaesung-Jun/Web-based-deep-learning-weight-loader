from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.conf import settings

from django.contrib import messages
from .upload_file_handler import handle_model_file, handle_weight_file

from .models import WeightFile, ModelFile
from .forms import WeightFileUploadForm , ModelFileUploadForm

from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.decorators import login_required

@login_required
def upload_file(request):
    
    storage = messages.get_messages(request)
    storage.used = True

    if request.method == 'POST':
        weight_file = WeightFile()
        model_file = ModelFile()
        
        weight_file.uploader = request.user
        model_file.uploader = request.user

        weight_file_form = WeightFileUploadForm(request.POST, request.FILES, instance=weight_file)
        model_file_form = ModelFileUploadForm(request.POST, request.FILES, instance=model_file)
        """
        if len(request.FILES['weight_file'].read()) > settings.MAX_UPLOAD_SIZE:
            messages.add_message(request, messages.ERROR, 'Please check max upload size')
        """
        if weight_file_form.is_valid() and model_file_form.is_valid():
            weight_file_form.save()
            model_file_form.save()

            model_file_objects = ModelFile.objects.exclude(id=weight_file.id)
            weight_file_objects = WeightFile.objects.exclude(id=model_file.id)
            
            print("Model file id = " + str(model_file_object.id) + " Deleted.")
            print("Weight file id = " + str(weight_file_object.id) + " Deleted.")
            
            
            messages.add_message(request, messages.SUCCESS, 'Upload Success!')
            return HttpResponseRedirect(reverse('upload-success'))

        elif not model_file_form.is_valid() and not weight_file_form.is_valid():
            messages.add_message(request, messages.ERROR, 'Weight File and Model File Upload Failed')
        elif not model_file_form.is_valid():
            messages.add_message(request, messages.ERROR, 'Model File Upload Failed')
        elif not weight_file_form.is_valid():
            messages.add_message(request, messages.ERROR, 'Weight File Upload Failed')

    weight_form = WeightFileUploadForm()
    model_form = ModelFileUploadForm()
    context = {
        'weight_form': weight_form,
        'model_form': model_form,
    }
    return render(request, 'weight_upload.html', context)

@login_required
def upload_success(request):
    return render(request, 'upload_success.html')