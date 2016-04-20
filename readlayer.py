import numpy

import theano
import theano.tensor as T
import pdb


class ReadLayer(object):

    def __init__(self, rng, input, input_shape, image, image_shape, N, irange=0.01, name='Default_focus'):
        print('Building layer: ' + name)
        self.input = input
        A = image_shape[0]
        B = image_shape[1]
        W_values = numpy.asarray(
            rng.uniform(
                low=-irange,
                high=irange,
                size=(5, input_shape[0])
            ),
            dtype=theano.config.floatX
        )
        self.W = theano.shared(value=W_values, name='W', borrow=True)
        b_values = numpy.zeros((5,1), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b', borrow=True)
        # initialize mean locations

        mu_x_values = numpy.zeros((N), dtype=theano.config.floatX)
        mu_x = theano.shared(value=mu_x_values, name='mu_x', borrow=True)
        mu_y_values = numpy.zeros((N), dtype=theano.config.floatX)
        mu_y = theano.shared(value=mu_y_values, name='mu_y', borrow=True)
        index_values = numpy.asarray(range(N), dtype=theano.config.floatX)
        mu_ind = theano.shared(value=index_values, name='mu_ind', borrow=True)
        A_ind_values = numpy.asarray(range(A), dtype=theano.config.floatX)
        A_ind = theano.shared(value=A_ind_values, name='A_ind', borrow=True)
        B_ind_values = numpy.asarray(range(B), dtype=theano.config.floatX)
        B_ind = theano.shared(value=B_ind_values, name='B_ind', borrow=True)

        lin_output = (T.dot(self.W, self.input) + self.b).flatten()

        self.g_x = g_x = ((A + 1) / 2)*(lin_output[0] + 1)
        self.g_y = g_y = ((B + 1) / 2)*(lin_output[1] + 1)
        self.delta = delta = (max(A, B) - 1) / (N - 1) * T.exp(lin_output[2])
        self.sigma_sq = sigma_sq = T.exp(lin_output[3])
        self.gamma = gamma = T.exp(lin_output[4])

        mu_x = g_x + (mu_ind - N / 2 - 0.5) * delta
        mu_y = g_y + (mu_ind - N / 2 - 0.5) * delta

        mu_x_ = mu_x.reshape((N, 1)).repeat(A, axis=1)
        A_ind_ = A_ind.reshape((1, A)).repeat(N, axis=0)
        mu_y_ = mu_y.reshape((N, 1)).repeat(B, axis=1)
        B_ind_ = B_ind.reshape((1, B)).repeat(N, axis=0)

        # Compute X filter banks##
        F_x = T.exp(-((A_ind_ - mu_x_)**2) / (2 * sigma_sq))
        F_x_div = T.sum(F_x, axis=1)
        F_x = (F_x.T / F_x_div).T

        # Compute Y filter banks##
        F_y = T.exp(-((B_ind_ - mu_y_)**2) / (2 * sigma_sq))
        F_y_div = T.sum(F_y, axis=1)
        F_y = (F_y.T / F_y_div).T

        self.glimpse = gamma * F_y.dot(image).dot(F_x.T)
        self.params = [self.W, self.b]
