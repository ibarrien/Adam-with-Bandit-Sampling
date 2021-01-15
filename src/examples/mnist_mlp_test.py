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


@editor: ibarrien
"""

from __future__ import print_function

import time
import argparse
import sys
sys.path.append('./')  # ideally, modules inside packages are not run as main...

from src.training import ImportanceTraining, BanditImportanceTraining


import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.optimizers import RMSprop
from keras.regularizers import l2

from importance_sampling.training import ImportanceTraining
from importance_sampling.training_bandit_package import BanditImportanceTraining
from example_utils import get_parser

if __name__ == "__main__":
    parser = get_parser("Train an MLP on MNIST")
    args = parser.parse_args()

    batch_size = 128
    num_classes = 10
    epochs = 10
    print('Epochs: %d' % epochs)

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

    model = Sequential()
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5),
                    input_shape=(784,)))
    model.add(Dense(512, activation='relu', kernel_regularizer=l2(1e-5)))
    model.add(Dense(10, kernel_regularizer=l2(1e-5)))
    model.add(Activation('softmax'))

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # get weights for samplers
    W = model.get_weights()
    model.set_weights(W)
    # Vanilla FIT
    base_start_time = time.time()
    model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
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
    history = wrapped.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    sampler_end_time = time.time()
    sampler_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', sampler_score[0])
    print('Test accuracy:', sampler_score[1])
    sampler_train_time = sampler_end_time - sampler_start_time
    sampler_score = model.evaluate(x_test, y_test, verbose=0)
    sampler_test_loss, sampler_test_acc = sampler_score[0], sampler_score[1]


    # Print the results
    print("Baseline test acc: ", base_test_acc, " in ", base_train_time, "s")
    print("Bandit test acc: ", sampler_test_acc, " in ", sampler_train_time, "s")
