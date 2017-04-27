#!/usr/bin/env python

import glob
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap
# construct cmap
flatui = ["#9b59b6", "#3498db", "#e74c3c", "#34495e", "#2ecc71"]
my_cmap = ListedColormap(sns.color_palette(flatui).as_hex())


for experiment_path in glob.glob("results/*"):

    print experiment_path
    epochs = glob.glob(experiment_path+"/*")

    filtered_number_of_epochs = []
    for _x in epochs:
        try:
            foo = int(_x.split("/")[-1])
            filtered_number_of_epochs.append(_x)
        except:
            pass
    number_of_epochs = len(filtered_number_of_epochs)


    for _epoch_number in range(number_of_epochs):
        _epoch_number = str(_epoch_number).zfill(2)#TO-DO: Make it generic
        projection = pickle.load(open(experiment_path+"/"+_epoch_number+"/projections.npy"))

        class_indices = projection['classIndex_list']
        class_map = projection['classMap']
        projections = projection['projections']

        X = []
        Y = []
        for projection in projections:
            X.append(projection[0])
            Y.append(projection[1])

        plt.clf()
        plt.scatter(X,Y, c=class_indices, cmap=my_cmap)
        plt.colorbar()
        plt.xlabel('Epoch : '+str(_epoch_number))
        plt.title("Projection for prediction of age of C.Elegans.")

        plt.savefig(experiment_path+"/"+_epoch_number+"/plot.png")
        print experiment_path, _epoch_number
