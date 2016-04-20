import numpy as np

import theano
import theano.tensor as T

import pdb


class LSTMLayer(object):

    def __init__(self, rng, h_tm1, c_tm1, input, n_out,
                 n_in=None, irange=0.01, name='Default_hidden',
                 C=None,
                 W=None, U=None, V=None, B=None,
                 ):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        print('Building layer: ' + name)
        self.input = input
        self.n_out = n_out
        self.n_in = n_in
        self.rng = rng
        self.irange = irange


        # [input, forget, cell, output]

        if W is None:
            W = [None] * 4
            for i in range(0, 4):
                W[i] = self.init_weights((n_out, n_in),
                                         name=("W_"+str(i)))
        if U is None:
            U = [None] * 4
            for i in range(0, 4):
                U[i] = self.init_weights((n_out, n_out),
                    name=("U_"+str(i)))

        # TODO: V should be diagonal as per graves 2014
        if V is None:
            V = [None] * 3
            for i in range(0, 3):
                V[i] = self.init_weights(
                    (n_out, 1), name=("V_"+str(i)))

        if B is None:
            B = [None] * 4
            for i in range(0, 4):
                b_values = np.zeros(
                    (n_out, 1), dtype=theano.config.floatX)
                B[i] = theano.shared(value=b_values, name=('b_'+str(i)), borrow=True)

        self.W = W
        self.U = U
        self.V = V
        self.B = B

        F_i = T.nnet.sigmoid(
            T.dot(self.W[0], self.input) +
            T.dot(self.U[0], h_tm1) +
            (self.V[0]*c_tm1) +
            self.B[0])
        F_f = T.nnet.sigmoid(
            T.dot(self.W[1], self.input) +
            T.dot(self.U[1], h_tm1) +
            (self.V[1]*c_tm1) +
            self.B[1])

        self.C = F_f * c_tm1 + F_i * T.tanh(
            T.dot(self.W[2], self.input) +
            T.dot(self.U[2], h_tm1) +
            self.B[2]
        )
        F_o = T.nnet.sigmoid(
            T.dot(self.W[3], self.input) +
            T.dot(self.U[3], h_tm1) +
            (self.V[2]*self.C) +
            self.B[3])

        self.output = F_o * T.tanh(self.C)
        self.output = self.output.T

        # parameters of the model
        self.params = self.W + self.U + self.B + self.V


    def init_weights(self, shape, name="W"):
        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.        if W is None:
        W_values = np.asarray(
            self.rng.uniform(
                low=-self.irange,
                high=self.irange,
                size=shape
            ),
            dtype=theano.config.floatX
        )

        return theano.shared(value=W_values, name=name, borrow=True)

    def get_initial_state(self):
        '''
        Returns initial values for the state and output
        '''
        c_0 = np.zeros((self.n_out, self.n_out))
        h_0 = np.zeros((self.n_out, self.n_out))

        return theano.shared(c_0), t.shared(h_0)
