#!/usr/bin/env python
import glob
import os
import shutil
import random

#im = skimage.io.imread

target_dir = "data/train_test_uncoagulated"


files = glob.glob("data/exploded_256x256/*/*.jpg")

train_test_split = 0.8

random.shuffle(files)

train = files[:int(train_test_split * len(files))]
test = files[int(train_test_split * len(files)):]

for _file in train:
    print _file
    _class = _file.split("/")[-2]
    _filename = _file.split("/")[-1]

    try:
        os.makedirs(target_dir+"/train/"+_class)
    except:
        pass

    shutil.copy(_file, target_dir+"/train/"+_class+"/"+_filename)

for _file in test:
    print _file
    _class = _file.split("/")[-2]
    _filename = _file.split("/")[-1]

    try:
        os.makedirs(target_dir+"/test/"+_class)
    except:
        pass

    shutil.copy(_file, target_dir+"/test/"+_class+"/"+_filename)
