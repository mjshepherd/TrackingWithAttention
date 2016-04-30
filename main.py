import sys
import theano
import theano.tensor as T
import numpy as np
import models
import matplotlib.pyplot as plt
import pickle

from load_mnist import load_mnist, pad_mnist, movie_mnist, batch_pad_mnist

learning_rate = 0.0001
n_epochs = 20
batch_size = 50
sequence_length = 10
repeat_style = 'still'
patience = 15
improvement_threshold = 1.005
report_freq = 100.

file_name = 'TESTmodel_t%s_bs%d_sl%d_ep%d.p' % (
    repeat_style, batch_size, sequence_length, n_epochs)

import pdb

sys.setrecursionlimit(30000)  # Needed for pickling


def build_model():

    model = models.TestLSTM(
        (100, 100), learning_rate=learning_rate, batch_size=batch_size)

    model.compile(train_batch_size=batch_size)

    return model


def train_model(model, train_images, train_targets):
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    # Keep track of statistics
    train_error = []

    num_images = len(train_images)

    # Initialize some variables
    old_loss = None
    improvement = 100

    epoch = 0
    done_looping = False

    # frame, tx, ty = pad_mnist(train_images[0])
    # frame = np.expand_dims(frame, axis=0).repeat(batch_size, axis=0)
    # frames = np.expand_dims(frame, axis=0).repeat(sequence_length, axis=0)
    # target = train_targets[0]

    while (epoch < n_epochs) and (not done_looping):
        epoch += 1
        print('### EPOCH %d ###' % epoch)
        num_correct = 0
        tot_loss = 0
        for i in range(0, num_images - batch_size + 1, batch_size):

            # Construct training batch
            #train_batch = np.expand_dims(train_images[0], axis=0).repeat(batch_size, axis=0)
            train_batch = train_images[i:i + batch_size] / 255.


            if repeat_style is 'still':
                x = y = np.ones((batch_size)) * 36
                train_batch, tx, ty = batch_pad_mnist(train_batch, out_dim=100)
                tx = np.expand_dims(tx, axis=0).repeat(
                    sequence_length, axis=0) + 14
                ty = np.expand_dims(ty, axis=0).repeat(
                    sequence_length, axis=0) + 14
                train_batch = np.expand_dims(train_batch, axis=1)
                train_batch = train_batch.repeat(sequence_length, axis=1)

            elif repeat_style is 'movie':
                movie_gen = movie_mnist(train_batch)
                plc = np.zeros((sequence_length, batch_size, 100, 100))
                tx = np.zeros((sequence_length, batch_size))
                ty = np.zeros((sequence_length, batch_size))

                for j in range(sequence_length):
                    plc[j], tx[j], ty[j] = movie_gen.next()
                train_batch = np.swapaxes(plc, 0, 1)
                tx += 14
                ty += 14

            # Construct training target
            target = train_targets[i:i + batch_size]
            #target = train_targets[i:i+batch_size]
            #target = np.expand_dims(target, axis=0).repeat(batch_size, axis=0)

            prediction, loss = model.train(train_batch, target, ty, tx)
            for j in range(batch_size):
                if np.argmax(prediction[j]) == np.argmax(target[j]):
                    num_correct += 1
            tot_loss = tot_loss + loss
            if i % report_freq == 0 and i > 0:
                percent_correct = 100 * num_correct / (report_freq)
                av_loss = tot_loss / report_freq * batch_size
                print('Examples seen: %d' % ((epoch - 1) * num_images + i))
                print('  Percent correct: %f' % (percent_correct))
                print('  Loss: %f' % (av_loss))
                tot_loss = 0
                num_correct = 0

            if old_loss is not None:
                improvement = old_loss / loss

            old_loss = loss
            if patience <= epoch and improvement < improvement_threshold:
                print('Breaking due to low improvement')
                done_looping = True
                break

    print('Optimization complete.')

    return train_error

if __name__ == '__main__':
    train_images, train_labels = load_mnist(dataset="training", path="mnist")

    # convert labels to one-hot encoding
    targets = np.zeros((len(train_labels), 10), dtype=np.uint8)
    for i, label in enumerate(train_labels):
        targets[i, label] = 1

    model = build_model()

    train_error = train_model(model,
                              train_images,
                              targets)

    pickle.dump(model, open(file_name, "wb"))
    pdb.set_trace()
