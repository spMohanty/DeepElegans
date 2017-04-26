#!/usr/bin/env python

import glob
import numpy as np

for _epoch_path in glob.glob("results/*/*"):
    print _epoch_path
