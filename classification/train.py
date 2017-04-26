#!/usr/bin/env python
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.models import Model
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, CSVLogger
from multi_gpu import multi_gpu
from custom_callbacks import CSVLoggerCustom

import shutil
import os

GPU_COUNT = 2
# path to the model weights files.
# weights_path = '../keras/examples/vgg16_weights.h5'
# top_model_weights_path = 'bottleneck_fc_model.h5'
# dimensions of our images.
img_width, img_height = 256, 256

# train_data_dir = '../data/train_test/train'
# validation_data_dir = '../data/train_test/test'

COAGULATED = True

if COAGULATED:
    train_data_dir = '../data/train_test/train'
    validation_data_dir = '../data/train_test/test'
else:
    train_data_dir = '../data/train_test_uncoagulated/train'
    validation_data_dir = '../data/train_test_uncoagulated/test'


nb_train_samples = 4155
nb_validation_samples = 1065
epochs = 30
batch_size = 32 * GPU_COUNT

# build the VGG16 network
base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(100, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(5, activation='softmax'))

# note that it is necessary to start with a fully-trained
# classifier, including the top classifier,
# in order to successfully do fine-tuning
# top_model.load_weights(top_model_weights_path)

# add the model on top of the convolutional base
# model.add(top_model)
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

if GPU_COUNT > 1:
    model = multi_gpu(model, GPU_COUNT)
# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
# for layer in model.layers[:15]:
#     layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.summary()

if COAGULATED:
    experiment_name = "results/coagulated"
else:
    experiment_name = "results/uncoagulated"    

try:
    shutil.rmtree(experiment_name)
except:
    pass

try:
    os.makedirs(experiment_name+"/snapshots")
except:
    pass

_csvLogger = CSVLoggerCustom(experiment_name+"/log.csv")
_checkpointer = ModelCheckpoint(filepath=experiment_name+"/snapshots/checkpoint-{epoch:02d}-{val_acc:.2f}-{val_loss:.2f}.hdf5", verbose=0, save_best_only=False)

# fine-tune the model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[_csvLogger, _checkpointer],
    verbose=1)

#WORKING
