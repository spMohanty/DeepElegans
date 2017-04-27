#!/usr/bin/env python

import glob
from PIL import Image
import os
import imageio


for folders in ["results/coagulated/", "results/uncoagulated/"]:
    frames = glob.glob(folders+"/*")
    filtered_number_of_frames = []
    for _x in frames:
        try:
            foo = int(_x.split("/")[-1])
            filtered_number_of_frames.append(_x)
        except:
            pass
    number_of_frames = len(filtered_number_of_frames)

    filenames = []
    for n in range(number_of_frames):
        epoch_name = str(n).zfill(2)
        if n ==0:
            for k in range(10):
                filenames.append(os.path.abspath(folders+"/"+epoch_name+"/plot.png"))
        else:
            filenames.append(os.path.abspath(folders+"/"+epoch_name+"/plot.png"))


    images = [imageio.imread(fn) for fn in filenames]

    target_filename = open(folders+"/animation.gif", "w")
    kargs = { 'duration': 0.3 }
    imageio.mimsave(target_filename, images, 'GIF', **kargs)
