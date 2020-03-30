from django.shortcuts import render
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.contrib.auth.mixins import LoginRequiredMixin

from django.contrib import messages
from .upload_file_handler import handle_model_file, handle_weight_file

from .models import WeightFile, ModelFile
from .forms import WeightFileUploadForm , ModelFileUploadForm

from django.utils.translation import ugettext_lazy as _
from django.contrib.auth.decorators import login_required
import os
import main.keras_loader as kl

def weightDetailView(request, pk):
    weight_file = get_object_or_404(WeightFile, pk=pk)
    return HttpResponseRedirect("../../uploads/{}".format(weight_file.weight_file))

def modelDetailView(request, pk):
    model_file = get_object_or_404(ModelFile, pk=pk)
    return HttpResponseRedirect("../../uploads/{}".format(model_file.model_file))

@login_required
def uploadFile(request):
    
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

            model_file_objects = ModelFile.objects.exclude(pk=model_file.pk)
            weight_file_objects = WeightFile.objects.exclude(pk=weight_file.pk)
            print("Uploader = ", str(weight_file.uploader))
            for model_files in model_file_objects: 
                print("Model file id : {0}(File : {1}) Deleted.".format(str(model_file.id), model_file.model_file))
                model_files.delete()

            for weight_files in weight_file_objects:
                print("Weight file id : {0}(File : {1}) Deleted.".format(str(weight_files.id), weight_file.weight_file))
                weight_files.delete()

            messages.add_message(request, messages.SUCCESS, 'Upload Success!')

            return HttpResponseRedirect(reverse('upload-success', args=(str(model_file.pk), str(weight_file.pk))))

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
def uploadSuccess(request, model_pk, weight_pk):

    model_file = get_object_or_404(ModelFile, pk=model_pk)
    weight_file = get_object_or_404(WeightFile, pk=weight_pk)

    context={
        'model_file' : model_file,
        'weight_file' : weight_file,
    }
    kl.load(str(model_file.model_file), str(weight_file.weight_file))
    return render(request, 'upload_success.html', context)