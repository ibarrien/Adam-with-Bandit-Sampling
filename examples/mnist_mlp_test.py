"""
1. Train a simple fully connected NN on the MNIST dataset.
2. Compare uniform batch sampling vs Bandit sampling.

tensorflow==1.13.1
keras==2.2.4

#
# Copyright (c) 2017 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

Banit Sampling paper:
https://arxiv.org/abs/2010.12986

HOW TO RUN:
Adam-with-Bandit-Sampling> python examples\mnist_mlp_test.py

@editor: ibarrien
"""

import os
import time
import argparse
import numpy as np
from matplotlib import pyplot as plt

from mnist_config import Config
from example_utils import get_parser, TimeHistory
import sys
sys.path.append('./')
from src.training import ImportanceTraining, BanditImportanceTraining

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

if __name__ == "__main__":
    parser = get_parser("Train an MLP on MNIST")
    args = parser.parse_args()
    epochs = Config.EPOCHS
    for batch_size in Config.BATCH_SIZE_LIST:
        num_classes = 10
        # the data, shuffled and split between train and test sets
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        print(x_train.shape[0], 'train samples')
        print(x_test.shape[0], 'test samples')

        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)
        # SET BASELINE MLP
        model = Sequential()
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5),
                        input_shape=(784,)))
        model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5)))
        model.add(Dense(10, kernel_regularizer=l2(1e-5)))
        model.add(Activation('softmax'))

        model.summary()

        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(Config.ADAM_LEARNING_RATE),
                      metrics=['accuracy'])

        # get weights for samplers
        W = model.get_weights()
        model.set_weights(W)
        # Vanilla FIT
        base_start_time = time.time()
        base_time_callback = TimeHistory()
        base_history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=Config.EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[base_time_callback]
        )
        base_loss_hist = base_history.history['loss']
        base_times = base_time_callback.times
        base_times_hist = np.cumsum(base_times)
        plt.plot(base_times_hist, base_loss_hist, '-*', label='Adam')
        print('Base Loss')
        print(base_loss_hist)
        print('Base times')
        print(base_times)
        base_end_time = time.time()
        base_train_time = base_end_time - base_start_time
        # VANILLA EVALUATE
        base_score = model.evaluate(x_test, y_test, verbose=0)
        base_test_loss, base_test_acc = base_score[0], base_score[1]

        # IMPORTANCE SAMPLER
        model.set_weights(W)
        args.importance_training = False
        args.bandit_training = True
        print('args importance training: ', args.importance_training)
        print('args bandit training: ', args.bandit_training)
        if args.importance_training:
            wrapped = ImportanceTraining(model, presample=5)
        elif args.bandit_training:
            wrapped = BanditImportanceTraining(model)
        else:
            wrapped = model
        # FIT
        sampler_start_time = time.time()
        bandit_time_callback = TimeHistory()
        bandit_history = wrapped.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=Config.EPOCHS,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[bandit_time_callback]
        )
        bandit_loss_hist = bandit_history.history['loss']
        bandit_times = bandit_time_callback.times
        bandit_times_hist = np.cumsum(bandit_times)
        plt.plot(bandit_times_hist, bandit_loss_hist, '-.', label='AdamBandit')
        plt.legend()
        plt.title('TrainBatchAvgLoss vs CumTime, epochs: %d' % Config.EPOCHS)
        plt.xlabel('Epoch time [s]')
        plt.ylabel('TrainLoss [batchSize=%d]' % batch_size)
        plot_save_path = os.path.join(Config.SAVE_PLOT_DIR, 'batch_size_%d.png' % batch_size)
        plt.savefig(plot_save_path)
        plt.close()
        sampler_end_time = time.time()
        sampler_score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', sampler_score[0])
        print('Test accuracy:', sampler_score[1])
        sampler_train_time = sampler_end_time - sampler_start_time
        sampler_score = model.evaluate(x_test, y_test, verbose=0)
        sampler_test_loss, sampler_test_acc = sampler_score[0], sampler_score[1]

        # Print the results
        print('Epochs: %d' % Config.EPOCHS)
        print('BatchSize: %d' % batch_size)
        print("Baseline test acc: ", base_test_acc, " in ", base_train_time, "s")
        print("Bandit test acc: ", sampler_test_acc, " in ", sampler_train_time, "s")
