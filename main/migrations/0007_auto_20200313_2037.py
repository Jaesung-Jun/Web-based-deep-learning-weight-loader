# Generated by Django 3.0.3 on 2020-03-13 11:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0006_auto_20200313_2028'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelfile',
            name='upload_time',
            field=models.TimeField(auto_now_add=True, null=True),
        ),
        migrations.AddField(
            model_name='weightfile',
            name='upload_time',
            field=models.TimeField(auto_now_add=True, null=True),
        ),
    ]
