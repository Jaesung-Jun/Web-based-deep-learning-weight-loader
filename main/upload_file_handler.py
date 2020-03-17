import os

def handle_weight_file(f):
    path = 'a.png'
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)

def handle_model_file(f):
    path = 'uploads/model_upload/'
    with open(path, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)