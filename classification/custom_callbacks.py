#!/usr/bin/env python

import os
import csv
import numpy as np

from collections import deque
from collections import OrderedDict
from collections import Iterable

from keras.callbacks import Callback

class CSVLoggerCustom(Callback):
    """Callback that streams epoch results to a csv file.
    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.
    # Example
        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```
    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        separator: string used to separate elements in the csv file.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, separator=',', append=False):
        self.sep = separator
        self.filename = filename
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.current_epoch = 0
        super(CSVLoggerCustom, self).__init__()

    def on_train_begin(self, logs=None):
        if self.append:
            if os.path.exists(self.filename):
                with open(self.filename) as f:
                    self.append_header = not bool(len(f.readline()))
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["epoch_"] = self.current_epoch
        logs["val_loss"] = -1000 #Ignore this value when parsing CSV
        logs["val_acc"] = -1000 #Ignore this value when parsing CSV
        logs['batch'] = batch

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k


        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        row_dict = OrderedDict({})


        row_dict.update((key, handle_value(logs[key])) for key in sorted(['batch','acc','epoch_','loss','val_acc','val_loss']))
        self.writer.writerow(row_dict)
        self.csv_file.flush()


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['batch'] = -1000 #Ignore this value when parsing CSV
        logs['epoch_'] = epoch

        self.current_epoch = epoch + 1

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, Iterable) and not is_zero_dim_ndarray:
                return '"[%s]"' % (', '.join(map(str, k)))
            else:
                return k

        if not self.writer:
            self.keys = sorted(logs.keys())

            class CustomDialect(csv.excel):
                delimiter = self.sep

            self.writer = csv.DictWriter(self.csv_file,
                                         fieldnames=self.keys, dialect=CustomDialect)
            if self.append_header:
                self.writer.writeheader()

        #row_dict = OrderedDict({'epoch': epoch})
        row_dict = OrderedDict({})

        row_dict.update((key, handle_value(logs[key])) for key in sorted(['batch','acc','epoch_','loss','val_acc','val_loss']))
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_train_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
