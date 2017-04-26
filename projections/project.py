#!/usr/bin/env python

import keras
from keras.layers import Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

from keras.layers.core import K
import tensorflow as tf
# from custom_generator import CustomImageDataGenerator

import numpy as np
import uuid

import scipy.misc

import glob
import os
import shutil

COAGULATED = False

if COAGULATED:
    PATH_TO_WEIGHTS = "/mount/SDF/DeepElegans/classification/results/coagulated-vgg16/snapshots"
    image_dir = "/mount/SDF/DeepElegans/data/train_test/test/"
    OUTPUT_DIRECTORY = "results/coagulated"
else:
    PATH_TO_WEIGHTS = "/mount/SDF/DeepElegans/classification/results/uncoagulated-vgg16/snapshots"
    image_dir = "/mount/SDF/DeepElegans/data/train_test_uncoagulated/test/"
    OUTPUT_DIRECTORY = "results/uncoagulated"


try:
    shutil.rmtree(OUTPUT_DIRECTORY)
except:
    pass


for _path_to_weights in glob.glob(PATH_TO_WEIGHTS+"/*.hdf5"):
    checkpoint_number = _path_to_weights.split("/")[-1].split("-")[1]
    MODEL_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY+"/"+checkpoint_number

    img_width, img_height = 256, 256
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        image_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')


    model = keras.models.load_model(_path_to_weights)

    # open("model.txt", "w").write(model.to_yaml())

    # exit(0)
    extractor = Model(inputs=model.layers[-2].layers[0].input, outputs=model.layers[-2].layers[-2].output)


    print "Model loaded...."

    classes = sorted(['2016.07.18', '2016.07.20', '2016.07.22', '2016.07.24', '2016.07.26'])
    TOTAL_TEST_SET_SIZE = len(glob.glob(image_dir+'/*/*'))

    count = 0
    total = 0
    for data in validation_generator:
        predictions = model.predict(data[0])
        predictions = [np.argmax(x) for x in predictions]
        true_labels = data[1]
        true_labels = [np.argmax(x) for x in true_labels]

        for _idx, val in enumerate(predictions):
            if predictions[_idx] == true_labels[_idx]:
                count += 1
            total += 1

            print "Accuracy : ", count*1.0/total
            print "Total : ", total

        #Extract Features

        #features = extractor.predict(np.array([data[0][0]]))
        features = extractor.predict(data[0])

        print features
        print features.shape

        for _idx, _feature in enumerate(features):
            image_id = str(uuid.uuid4())
            target_dir_path = MODEL_OUTPUT_DIRECTORY+"/"+classes[true_labels[_idx]]+"/"+image_id
            try:
                os.makedirs(target_dir_path)
            except:
                pass

            # Save Image first
            scipy.misc.imsave(target_dir_path+"/image.jpg", data[0][_idx]*255)
            # Save Extracted Features
            np.save(target_dir_path+"/block5_pool_features.npy", features[_idx])
            print "Shape : ",features[_idx].shape
        break

        if total > TOTAL_TEST_SET_SIZE:
            break
