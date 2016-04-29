import os
import struct
from array import array as pyarray
import numpy as np
import theano
from numpy import array, int8, uint8, zeros
import math
import pdb


def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'images-train')
        fname_lbl = os.path.join(path, 'labels-train')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'images-test')
        fname_lbl = os.path.join(path, 'labels-test')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    ind = [k for k in range(size) if lbl[k] in digits]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(
            img[ind[i] * rows * cols: (ind[i] + 1) * rows * cols]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels


def batch_pad_mnist(images, out_dim=100, x=None, y=None):
    result = np.zeros((images.shape[0], out_dim, out_dim))
    if x is not None and y is not None:
        for i, image in enumerate(images):
            result[i], x[i], y[i] = pad_mnist(image, out_dim=out_dim, x=x[i], y=y[i])
    else:
        x = np.zeros((images.shape[0]))
        y = np.zeros((images.shape[0]))
        for i, image in enumerate(images):
            result[i], x[i], y[i] = pad_mnist(image, out_dim=out_dim)
    return result, x, y

def pad_mnist(image, out_dim=100, x=None, y=None):
    
    if x is None:
        x = np.ceil(np.random.random() * (out_dim - image.shape[1]))
    if y is None:
        y = np.ceil(np.random.random() * (out_dim - image.shape[0]))

    x = x.astype(int)
    y = y.astype(int)

    result = np.zeros((out_dim, out_dim), dtype=theano.config.floatX)
    result[y:(y + image.shape[1]), x:(x + image.shape[0])] = image

    return result, x, y


def movie_mnist(image, out_dim=100, speed=10, variation=2):
    batch_size = image.shape[0]
    speed = np.random.random((batch_size)) * variation + speed
    direction = np.random.random((batch_size)) * np.pi * 2
    dX = np.floor(np.cos(direction) * speed)
    dY = np.floor(np.sin(direction) * speed)
    effectiveX = (out_dim - image.shape[2])
    effectiveY = (out_dim - image.shape[1])

    X = np.floor(np.random.random((batch_size)) * effectiveX)
    Y = np.floor(np.random.random((batch_size)) * effectiveY)
    while True:
        dX[X+dX > (effectiveX-1)] = -dX[X+dX > (effectiveX-1)]
        dX[X+dX < 0] = -dX[X+dX < 0]

        dY[Y+dY > (effectiveY-1)] = -dY[Y+dY > (effectiveY-1)]
        dY[Y+dY < 0] = -dY[Y+dY < 0]

        X = X + dX
        Y = Y + dY
        yield(batch_pad_mnist(image, out_dim, x=X, y=Y))
