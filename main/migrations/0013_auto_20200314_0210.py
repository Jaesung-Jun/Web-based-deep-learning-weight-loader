# Generated by Django 3.0.3 on 2020-03-13 17:10

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0012_auto_20200314_0204'),
    ]

    operations = [
        migrations.AlterField(
            model_name='modelfile',
            name='id',
            field=models.UUIDField(default=uuid.uuid1, help_text='Unique ID for model file.', primary_key=True, serialize=False, unique=True),
        ),
        migrations.AlterField(
            model_name='weightfile',
            name='id',
            field=models.UUIDField(default=uuid.uuid1, help_text='Unique ID for weight file.', primary_key=True, serialize=False, unique=True),
        ),
    ]