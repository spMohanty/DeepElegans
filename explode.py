#!/usr/bin/env python
import glob
import skimage.io
from skimage.transform import resize
import os

#im = skimage.io.imread

for _file in glob.glob("data/RGB/*/*.tif"):
    im = skimage.io.imread(_file)
    print "="*80
    for i in range(15):
        sub_im = im[0]
        _class = _file.split("/")[-2]
        _filename = ".".join(_file.split("/")[-1].split(".")[:-1])
        print _class, _filename
        print str(i)+"/15"
        try:
            os.mkdir("data/exploded/"+_class)
        except:
            pass
        try:
            os.mkdir("data/exploded_256x256/"+_class)
        except:
            pass
        skimage.io.imsave("data/exploded/"+_class+"/"+_filename+"___"+str(i)+".jpg", sub_im)

        resized_image = resize(sub_im, (256, 256))
        skimage.io.imsave("data/exploded_256x256/"+_class+"/"+_filename+"___"+str(i)+".jpg", resized_image)
