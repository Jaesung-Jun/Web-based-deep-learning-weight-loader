from django.contrib import admin
from main.models import WeightFile, ModelFile 
# Register your models here.

@admin.register(WeightFile)
class WeightFileAdmin(admin.ModelAdmin):
    list_display = ('upload_date', 'upload_time', 'uploader', 'weight_file', 'id')

@admin.register(ModelFile)
class ModelFileAdmin(admin.ModelAdmin):
    list_display = ('upload_date', 'upload_time', 'uploader', 'model_file', 'id')

# admin.site.register(ModelFile)
# admin.site.register(WeightFile)