import numpy as np

import theano
import theano.tensor as T

import pdb


class LSTMLayer(object):

    def __init__(self, rng, n_out, n_in=None,
                 irange=0.01, name='Default_hidden',
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
        print('Initializing layer: ' + name)
        self.name = name
        self.n_out = n_out
        self.n_in = n_in
        self.rng = rng
        self.irange = irange

        # [input, forget, cell, output]

        if W is None:
            W = [None] * 4
            for i in range(0, 4):
                W[i] = self.init_weights((n_in, n_out),
                                         name=("W_" + str(i)))
        if U is None:
            U = [None] * 4
            for i in range(0, 4):
                U[i] = self.init_weights((n_out, n_out),
                                         name=("U_" + str(i)))

        # TODO: V should be diagonal as per graves 2014
        if V is None:
            V = [None] * 3
            for i in range(0, 3):
                V[i] = self.init_weights(
                    (n_out), name=("V_" + str(i)))

        if B is None:
            B = [None] * 4
            for i in range(0, 4):
                b_values = np.zeros(
                    (n_out), dtype=theano.config.floatX)
                B[i] = theano.shared(
                    value=b_values, name=('b_' + str(i)), borrow=True)

        self.W = W
        self.U = U
        self.V = V
        self.B = B
        # parameters of the layer
        self.params = self.W + self.U + self.B + self.V

    def one_step(self, input, h_tm1, c_tm1):
        '''
        input: batches, X
        h_tmi: batches, self.n_out
        c_tmi: batches, self.n_out
        '''
        F_i = T.nnet.sigmoid(
            T.dot(input, self.W[0]) +
            T.dot(h_tm1, self.U[0]) +
            (c_tm1 * self.V[0].dimshuffle(['x', 0])) +
            self.B[0].dimshuffle(['x', 0]))
        F_f = T.nnet.sigmoid(
            T.dot(input, self.W[1]) +
            T.dot(h_tm1, self.U[1]) +
            (c_tm1 * self.V[1].dimshuffle(['x', 0])) +
            self.B[1].dimshuffle(['x', 0]))

        c = F_f * c_tm1 + F_i * T.tanh(
            T.dot(input, self.W[2]) +
            T.dot(h_tm1, self.U[2]) +
            self.B[2].dimshuffle(['x', 0])
        )
        F_o = T.nnet.sigmoid(
            T.dot(input, self.W[3]) +
            T.dot(h_tm1, self.U[3]) +
            (c * self.V[2].dimshuffle(['x', 0])) +
            self.B[3].dimshuffle(['x', 0]))
        h = F_o * T.tanh(c)

        return h, c

    def init_weights(self, shape, name="W"):
        W_values = np.asarray(
            self.rng.uniform(
                low=-self.irange,
                high=self.irange,
                size=shape
            ),
            dtype=theano.config.floatX
        )

        return theano.shared(value=W_values, name=name, borrow=True)

    def get_initial_state(self, batch_size):
        '''
        Returns initial values for the state and output
        '''
        c_0 = np.zeros((batch_size, self.n_out))
        h_0 = np.zeros((batch_size, self.n_out))

        return theano.shared(c_0, borrow=True), theano.shared(h_0, borrow=True)

if __name__ == "__main__":
    # testing
    rng = np.random.RandomState(23455)
    batch_size = 10
    n_out = 10
    n_in = 144

    layer = LSTMLayer(
        rng=rng,
        n_out=n_out,
        n_in=n_in)

    plc_in = T.matrix()
    plc_h = T.matrix()
    plc_c = T.matrix()

    h_out, c_out = layer.one_step(plc_in, plc_h, plc_c)

    one_step = theano.function(inputs=[plc_in, plc_h, plc_c],
                               outputs=[h_out, c_out],
                               allow_input_downcast=True)

    test_in = np.random.random((batch_size, n_in))
    h0 = np.random.random((batch_size, n_out))
    c0 = np.random.random((batch_size, n_out))

    res_h, res_c = one_step(test_in, h0, c0)
    assert res_h.shape == (batch_size, n_out), "Incorrect output shape"
    assert res_c.shape == (batch_size, n_out), "Incorrect state shape"
    print('Test complete')
