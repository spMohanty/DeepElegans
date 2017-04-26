#!/usr/bin/env python
import glob
import os
import shutil
import random

#im = skimage.io.imread

target_dir = "data/train_test"

print "class, key, filepath"
data = {} # Levels : _class, file_key, path
for _filepath in glob.glob("data/exploded_256x256/*/*.jpg"):
    _key = _filepath.split("/")[-1].split("___")[0]
    _class = _filepath.split("/")[-2]
    #data.append((_class, _key, _filepath))
    try:
        foo = data[_class]
    except:
        data[_class]={}

    try:
        foo = data[_class][_key]
    except:
        data[_class][_key]=[]

    data[_class][_key].append(_filepath)


# Train_Test_Split
train_test_split = 0.8

for _class in data.keys():
    file_keys = data[_class].keys()
    random.shuffle(file_keys)
    _train = file_keys[:int(train_test_split*len(file_keys))]
    _test = file_keys[int(train_test_split*len(file_keys)):]

    # _train = [data[_class][_x] for _x in _train]
    # _test = [data[_class][_x] for _x in _test]


    def arrange_train_test(data, _class, _obj, name):
        global target_dir
        for _file_key in _obj:
            for _filepath in data[_class][_file_key]:
                if not os.path.exists(target_dir+"/"+name+"/"+_class):
                    os.makedirs(target_dir+"/"+name+"/"+_class)
                shutil.copy(_filepath, target_dir+"/"+name+"/"+_class+"/"+_filepath.split("/")[-1])
                # print "Copying ", _filepath " ----> " , target_dir+"/"+name+"/"+_class+"/"+_filepath.split("/")[-1]
                print _class, _filepath
                # Copy into target direcotry

    # print _class, len(file_keys)
    arrange_train_test(data, _class, _train, 'train')
    arrange_train_test(data, _class, _test, 'test')
