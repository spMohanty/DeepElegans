#!/usr/bin/env python

import glob
import numpy as np
from sklearn.manifold import TSNE
import pickle
from tsne import bh_sne


for COAGULATED in [True, False]:
    classes = sorted(['2016.07.18', '2016.07.20', '2016.07.22', '2016.07.24', '2016.07.26'])
    classMap = {}
    for _idx, className in enumerate(classes):
        classMap[className] = _idx

    print classMap

    if COAGULATED:
        EPOCH_PATH = "/mount/SDF/DeepElegans/projections/results/coagulated/*"
    else:
        EPOCH_PATH = "/mount/SDF/DeepElegans/projections/results/uncoagulated/*"


    for epoch_path in glob.glob(EPOCH_PATH):
        print "epoch : ", epoch_path

        epoch = int(epoch_path.split("/")[-1])

        classIndex_list = []
        features_list = []

        for data_path in glob.glob(epoch_path+"/*/*"):
            className = data_path.split("/")[-2]
            classIndex_list.append(classMap[className])

            features = np.load(data_path+"/block5_pool_features.npy")
            # perform a pooling across axis 0 and 1 to reduces features from
            # shape (8, 8, 512) to (512,)
            features = np.mean(features, axis=(0,1)).tolist()
            features_list.append(features)

        # model = TSNE(n_components=2, random_state=0, n_iter=5000, verbose=2)
        # model.fit(features_list)
        # print "Length : ", len(features_list)

        # projections = model.fit_transform(features_list)
        projections = bh_sne(np.array(features_list), perplexity=41)


        X = []
        Y = []
        for projection in projections:
            X.append(projection[0])
            Y.append(projection[1])

        combined_results = {}
        combined_results['projections'] = projections
        combined_results['features_list'] = features_list
        combined_results['classIndex_list'] = classIndex_list
        combined_results['classMap'] = classMap

        pickle.dump(combined_results, open(epoch_path+"/projections.pickle", "wb"))
