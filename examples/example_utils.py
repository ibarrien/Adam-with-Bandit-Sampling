

import time
import argparse
from keras.callbacks import Callback


def get_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--uniform",
        action="store_false",
        dest="importance_training",
        help="Enable uniform sampling"
    )
    return parser

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
