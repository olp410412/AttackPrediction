import pandas as pd, numpy as np, os, gc
import time
import math

from keras import callbacks
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam



def process_line(line):
    tmp = [str(val) for val in line.strip().split(',')]

    x = np.array(tmp[:-1])

    y = np.array(tmp[-1:])

    return x, y


def generate_arrays_from_file(path, batch_size):
    while 1:

        f = open(path)

        cnt = 0

        X = []

        Y = []

        for line in f:

            # create Numpy arrays of input data

            # and labels, from each line in the file

            x, y = process_line(line)

            X.append(x)

            Y.append(y)

            cnt += 1

            if cnt == batch_size:
                cnt = 0

                yield (np.array(X), np.array(Y))

                X = []

                Y = []

    f.close()

if __name__ == '__main__':
    model = Sequential()
    model.add(Dense(100, input_dim=371))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Dropout(0.4))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer=Adam(lr=0.01), loss="binary_crossentropy", metrics=["accuracy"])
    annealer = LearningRateScheduler(lambda x: 1e-2 * 0.95 ** x)

    model.fit_generator(generate_arrays_from_file('F:\\temp\\buffer1.csv', batch_size=64),epochs=5,
                                                   samples_per_epoch=25024, max_q_size=1000, verbose=2, nb_worker=1)


