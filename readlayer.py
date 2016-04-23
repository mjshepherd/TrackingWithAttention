import numpy

import theano
import theano.tensor as T
from hiddenlayer import HiddenLayer
from reader import Reader


class ReadLayer(object):

    def __init__(self, rng, h_shape, image_shape, N, name='Default_readlayer'):
        print('Building layer: ' + name)

        self.lin_transform = HiddenLayer(
            rng,
            n_in=h_shape[0] * h_shape[1],
            n_out=5,
            activation=None,
            name='readlayer: linear transformation')

        self.reader = Reader(
            rng,
            image_shape=image_shape,
            N=N,
            name='readlayer: reader')

        self.params = self.lin_transform.params

    def one_step(self, h, image):
        linear = self.lin_transform.one_step(h)
        read, g_x, g_y, delta, sigma_sq = self.reader.one_step(linear, image)
        return read, g_x, g_y, delta, sigma_sq
