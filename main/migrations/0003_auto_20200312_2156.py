# Generated by Django 3.0.3 on 2020-03-12 12:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_auto_20200312_2145'),
    ]

    operations = [
        migrations.AlterField(
            model_name='weightfile',
            name='weight_file',
            field=models.FileField(upload_to='weight_upload/'),
        ),
    ]