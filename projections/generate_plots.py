#!/usr/bin/env python

import glob
import numpy as np
import pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

for experiment_path in glob.glob("results/*"):
    print experiment_path
    number_of_epochs = len(glob.glob(experiment_path+"/*"))
    for _epoch_number in range(number_of_epochs):
        _epoch_number = str(_epoch_number).zfill(2)#TO-DO: Make it generic
        projection = pickle.load(open(experiment_path+"/"+_epoch_number+"/projections.pickle"))

        class_indices = projection['classIndex_list']
        class_map = projection['classMap']
        projections = projection['projections']

        X = []
        Y = []
        for projection in projections:
            X.append(projection[0])
            Y.append(projection[1])

        plt.clf()
        plt.scatter(X,Y, c=class_indices)
        plt.legend(('2016.07.18', '2016.07.20', '2016.07.22', '2016.07.24', '2016.07.26'), loc="lower right")
        plt.savefig(experiment_path+"/"+_epoch_number+"/plot.png")
        print experiment_path, _epoch_number
