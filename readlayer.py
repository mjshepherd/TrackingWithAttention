import numpy

import theano
import theano.tensor as T
from hiddenlayer import HiddenLayer
from reader import Reader


class ReadLayer(object):

    def __init__(self, rng, h, h_shape, image, image_shape, N, name='Default_readlayer'):
        print('Building layer: ' + name)

        self.lin_transform = HiddenLayer(
            rng,
            input=h.flatten(),
            n_in=h_shape[0] * h_shape[1],
            n_out=5,
            activation=None,
            name='readlayer: linear transformation')

        self.reader = Reader(
            rng,
            l=self.lin_transform.output,
            image=image,
            image_shape=image_shape,
            N=N,
            name='readlayer: reader')

        self.output = self.reader.glimpse
        self.params = self.lin_transform.params
