# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time
from keras.datasets import cifar10
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.engine import Input, Model
from keras.layers import Concatenate
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler, ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from datetime import datetime
import keras.backend as K
import json
import DataPreparation

nb_classes = 4

# the data, shuffled and split between train and test sets
dir = './Dataset(4classes)'

label = ("backchul", "bokryung", "gamcho", "insam")

(x_train, y_train), (x_test, y_test), (x_val, y_val) = DataPreparation.Data_All_Preprocessing(dir=dir, label=label, X_size=224, Y_size=224)

weight_save = False

# we reduce # filters by factor of 8 compared to original inception-v4
nb_filters_reduction_factor = 8

#time measure
start_time = time.time()

def inception_v4_stem(x):
    # in original inception-v4, conv stride is 2
    x = Convolution2D(32//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    x = Convolution2D(32//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    x = Convolution2D(64//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    
    # in original inception-v4, stride is 2
    a = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
    # in original inception-v4, conv stride is 2
    b = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    # x = merge([a, b], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b])
    
    a = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    a = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(a)
    b = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = Convolution2D(64//nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = Convolution2D(64//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(b)
    b = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(b)
    #x = merge([a, b], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b])
    
    # in original inception-v4, conv stride should be 2
    a = Convolution2D(192//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    # in original inception-v4, stride is 2
    b = MaxPooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
    
    #x = merge([a, b], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b])
    
    return x


def inception_v4_A(x):
    a = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering='tf')(x)
    a = Convolution2D(96//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(a)
    
    b = Convolution2D(96//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    
    c = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    
    d = Convolution2D(64//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    d = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = Convolution2D(96//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    
    #x = merge([a, b, c, d], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b, c, d])
    
    return x


def inception_v4_reduction_A(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
    b = Convolution2D(384//nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(x)
    c = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = Convolution2D(224//nb_filters_reduction_factor, 3, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = Convolution2D(256//nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(c)
    
    #x = merge([a, b, c], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b, c])
    
    return x
    

def inception_v4_B(x):
    a = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering='tf')(x)
    a = Convolution2D(128//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(a)
    
    b = Convolution2D(384//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    
    c = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = Convolution2D(224//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = Convolution2D(256//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    
    d = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    d = Convolution2D(192//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = Convolution2D(224//nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = Convolution2D(224//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = Convolution2D(256//nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    
    #x = merge([a, b, c, d], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b, c, d])
    
    return x


def inception_v4_reduction_B(x):
    a = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid', dim_ordering='tf')(x)
    b = Convolution2D(192//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    b = Convolution2D(192//nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(b)
    c = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c = Convolution2D(256//nb_filters_reduction_factor, 1, 7, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = Convolution2D(320//nb_filters_reduction_factor, 7, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c = Convolution2D(320//nb_filters_reduction_factor, 3, 3, subsample=(2, 2), activation='relu',
                      init='he_normal', border_mode='valid', dim_ordering='tf')(c)
    
    #x = merge([a, b, c], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b, c])
    
    return x


def inception_v4_C(x):
    a = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same', dim_ordering='tf')(x)
    a = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(a)
    
    b = Convolution2D(256//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    
    c = Convolution2D(384//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    c1 = Convolution2D(256//nb_filters_reduction_factor, 1, 3, subsample=(1, 1), activation='relu',
                       init='he_normal', border_mode='same', dim_ordering='tf')(c)
    c2 = Convolution2D(256//nb_filters_reduction_factor, 3, 1, subsample=(1, 1), activation='relu',
                       init='he_normal', border_mode='same', dim_ordering='tf')(c)
    
    d = Convolution2D(384//nb_filters_reduction_factor, 1, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(x)
    d = Convolution2D(448//nb_filters_reduction_factor, 1, 3, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d = Convolution2D(512//nb_filters_reduction_factor, 3, 1, subsample=(1, 1), activation='relu',
                      init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d1 = Convolution2D(256//nb_filters_reduction_factor, 3, 1, subsample=(1, 1), activation='relu',
                       init='he_normal', border_mode='same', dim_ordering='tf')(d)
    d2 = Convolution2D(256//nb_filters_reduction_factor, 1, 3, subsample=(1, 1), activation='relu',
                       init='he_normal', border_mode='same', dim_ordering='tf')(d)
    
    # x = merge([a, b, c1, c2, d1, d2], mode='concat', concat_axis=-1)
    x = Concatenate()([a, b, c1, c2, d1, d2])
    
    return x
 
img_rows, img_cols = 224, 224
img_channels = 3
 
# in original inception-v4, these are 4, 7, 3, respectively
num_A_blocks = 4
num_B_blocks = 7
num_C_blocks = 3
 
inputs = Input(shape=(img_rows, img_cols, img_channels))

x = inception_v4_stem(inputs)
for i in range(num_A_blocks):
    x = inception_v4_A(x)
x = inception_v4_reduction_A(x)
for i in range(num_B_blocks):
    x = inception_v4_B(x)
x = inception_v4_reduction_B(x)
for i in range(num_C_blocks):
    x = inception_v4_C(x)

x = AveragePooling2D(pool_size=(4, 4), strides=(1, 1), border_mode='valid', dim_ordering='tf')(x)
x = Dropout(0.5)(x)
x = Flatten()(x)

predictions = Dense(nb_classes, activation='softmax')(x)

model = Model(input=inputs, output=predictions)
model.summary()

# Commented out IPython magic to ensure Python compatibility.

model.compile(optimizer='adam',
               loss='categorical_crossentropy',
               metrics=['accuracy'])

batch_size = 16
nb_epoch = 1
data_augmentation = True

# Model saving callback
checkpointer = ModelCheckpoint(filepath='stochastic_depth_20190919.hdf5', verbose=1, save_best_only=True)

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(x_train, y_train, 
                        batch_size=batch_size, nb_epoch=nb_epoch, verbose=1,
                        validation_data=(x_test, y_test), shuffle=True,
                        callbacks=[])
else:
    print('Using real-time data augmentation.')

    # realtime data augmentation
    datagen_train = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True,
        vertical_flip=False)
    datagen_train.fit(x_train)

    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size, shuffle=True),
                                  samples_per_epoch=x_train.shape[0], 
                                  nb_epoch=nb_epoch, verbose=1,
                                  validation_data=(x_test, y_test),
                                  callbacks=[])


#time measure
end_time = time.time()

print("Elapsed Time : {} sec".format(end_time-start_time))

if weight_save == True:
    now_date = datetime.today().strftime("%Y%m%d%H%M%S")
    
    h5_file_name = "h5/inceptionv4_{}_save[{}].h5".format(now_date, (end_time-start_time))

    model.save_weights(h5_file_name)
    
    print("Saved model to disk - {}".format(h5_file_name))

scores = model.evaluate(x_val, y_val, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='best')

#loss_ax.legend(bbox_to_anchor=(0., 1.02, 1., .100), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')

acc_ax.legend(loc='best')

#acc_ax.legend(bbox_to_anchor=(0., 1.05, 1., .100), loc='lower left', ncol=2, mode="expand", borderaxespad=0.)

plt.show()
