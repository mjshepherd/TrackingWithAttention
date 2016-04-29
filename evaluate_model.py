import sys
import theano
import numpy as np
import models
import matplotlib.pyplot as plt
import pickle
import pdb
from load_mnist import load_mnist, pad_mnist, movie_mnist, batch_pad_mnist

sequence_length = 10
batch_size = 1
assert(len(sys.argv) == 3)

model_name = sys.argv[1]
sequence_type = sys.argv[2]


def main():
    print('Loading Model...')
    model = load_model(model_name)
    print('Loading Data...')
    images, labels = load_mnist(dataset="testing", path="mnist")
    images = images / 255.

    total_images = images.shape[0]
    print('Begining Tests...')
    # Initialize statistics placeholders
    correct_stats = np.zeros((total_images, sequence_length)) * -1
    IOU_stats = np.zeros((total_images, sequence_length)) * -1
    dist_stats = np.zeros((total_images, sequence_length)) * -1
    for i in range(0, total_images, batch_size):
        if i % 250 == 0:
            print('On example %d' % i)
        batch_images = images[i:i + batch_size]
        batch_labels = labels[i:i + batch_size]

        # generate test sequences
        if sequence_type == 'movie':
            movie_gen = movie_mnist(batch_images)
            plc = np.zeros((sequence_length, batch_size, 100, 100))
            tx = np.zeros((sequence_length, batch_size))
            ty = np.zeros((sequence_length, batch_size))

            for j in range(sequence_length):
                plc[j], tx[j], ty[j] = movie_gen.next()
            batch_images = np.swapaxes(plc, 0, 1)
            tx += 14
            ty += 14
        elif sequence_type == 'still':
            x = y = np.ones((batch_size)) * 36
            batch_images, tx, ty = batch_pad_mnist(batch_images, out_dim=100)
            tx = np.expand_dims(tx, axis=0).repeat(
                sequence_length, axis=0) + 14
            ty = np.expand_dims(ty, axis=0).repeat(
                sequence_length, axis=0) + 14
            batch_images = np.expand_dims(batch_images, axis=1)
            batch_images = batch_images.repeat(sequence_length, axis=1)
        # Now we loop for all time-steps
        reset = True  # reset is true for the first timestep
        for t in range(sequence_length):
            # OK now make a prediction
            prediction, attention_params = model.predict(batch_images[:, t, :, :], reset=reset)
            reset = False
            prediction = np.argmax(prediction, axis=1)

            correct_stats[i:i + batch_size, t] = prediction == batch_labels
            px = attention_params[1]
            py = attention_params[2]
            pdel = attention_params[3] * 6 # half the number of gaussians
            distance = get_dist(tx[t], ty[t], px, py)
            dist_stats[i:i + batch_size, t] = distance
            IOU_stats[i:i + batch_size, t] = get_IOU(tx[t], ty[t], 14, px, py, pdel)
    mean_dist = dist_stats.mean(axis=0)
    mean_correct = correct_stats.mean(axis=0)
    mean_IOU = IOU_stats.mean(axis=0)

    print(str(mean_dist))
    print(str(mean_correct))
    print(str(mean_IOU))


def get_dist(tx, ty, px, py):
    return np.sqrt((tx - px)**2 + (ty - py)**2)


def get_IOU(tx, ty, tdel, px, py, pdel):
    txl = tx - tdel
    txr = tx + tdel
    tyt = ty - tdel
    tyb = ty + tdel
    pxl = px - pdel
    pxr = px + pdel
    pyt = py - pdel
    pyb = py + pdel

    x_overlap = np.maximum(0, np.minimum(txr, pxr) - np.maximum(txl, pxl))
    y_overlap = np.maximum(0, np.minimum(tyb, pyb) - np.maximum(tyt, pyt))
    intersection = x_overlap * y_overlap
    union = (2. * tdel)**2 + (2. * pdel)**2 - intersection
    return intersection / union


def load_model(name):
    return pickle.load(open(name, 'r'))

if __name__ == '__main__':
    main()
