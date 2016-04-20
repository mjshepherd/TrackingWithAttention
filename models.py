import numpy as np
import theano
import theano.tensor as T
from hiddenlayer import HiddenLayer
from convpoollayer import ConvPoolLayer
from LSTMLayer import LSTMLayer
from readlayer import ReadLayer

import pdb

rng = np.random.RandomState(23455)


class AbstractModel():
    def get_grads(self):
        return T.grad(self.loss, self.params)

    def compile(self, updates):
        print("Compiling function")
        self.predict_func = theano.function(
            inputs=[self.input], outputs=[self.output])
        self.train_func = theano.function(
            inputs=[self.input, self.target],
            outputs=[self.output, self.loss],
            updates=updates
        )
        print("Done!")

    def set_loss(self, loss):
        self.loss = loss

    def train(self, x, y):
        return self.train_func(x, y)

    def predict(self, x):
        return self.predict_func(x)


class TestModel(AbstractModel):

    def __init__(self, input, input_dims, target):
        self.input = input
        self.target = target
        num_in = input_dims[0] * input_dims[1]
        layer1 = HiddenLayer(
            rng,
            input=input.flatten(),
            n_in=num_in,
            n_out=10,
            activation=T.tanh,
            name='FC1'
        )
        layer2 = HiddenLayer(
            rng,
            input=layer1.output,
            n_in=10,
            n_out=10,
            activation=T.tanh,
            name='output'
        )
        self.output = T.nnet.softmax(layer2.output)
        self.params = layer1.params + layer2.params


class TestConvModel(AbstractModel):

    def __init__(self, input, input_dims, target):
        self.input = input
        self.target = target
        conv_compat_shape = (1, 1, input_dims[0], input_dims[1])
        conv_input = self.input.reshape(conv_compat_shape)

        layer1 = ConvPoolLayer(
            rng,
            input=conv_input,
            name='C1',
            filter_shape=(20, 1, 3, 3),
            input_shape=conv_compat_shape,
            poolsize=(2, 2)
        )

        layer2 = ConvPoolLayer(
            rng,
            input=layer1.output,
            name='C2',
            filter_shape=(20, 20, 3, 3),
            input_shape=layer1.output_shape,
            poolsize=(2, 2)
        )

        layer3 = HiddenLayer(
            rng,
            input=layer2.output.flatten(ndim=2),
            n_in=reduce(lambda x, y: x * y, layer2.output_shape),
            n_out=10,
            activation=T.tanh,
            name='output')
        self.output = T.nnet.softmax(layer3.output)
        self.params = layer1.params + layer2.params + layer3.params


class TestLSTM(AbstractModel):

    def __init__(self, input, input_dims, target):
        self.input = input
        self.target = target
        self.h_tm1 = T.matrix(name="hidden_output", dtype=theano.config.floatX)
        self.c_tm1 = T.matrix(name="hidden_state", dtype=theano.config.floatX)
        self.hidden_output = None
        self.hidden_state = None
        num_in = input_dims[0] * input_dims[1]

        self.lstm_layer_sizes = [100]
        layer_outputs = self.deserialize(self.h_tm1)
        layer_states = self.deserialize(self.c_tm1)
        read = ReadLayer(
            rng,
            input=layer_outputs[0],
            input_shape=(self.lstm_layer_sizes[0], 1),
            image=self.input,
            image_shape=input_dims,
            N=12,
            name='read'
        )

        layer1 = LSTMLayer(
            rng,
            input=read.glimpse.reshape((144, 1)),
            # input=self.input.reshape((num_in, 1)),
            n_in=144,
            c_tm1=layer_outputs[0],
            h_tm1=layer_states[0],
            n_out=self.lstm_layer_sizes[0],
            name='LSTM1'
        )
        layer2 = HiddenLayer(
            rng,
            input=layer1.output,
            n_in=100,
            n_out=10,
            activation=T.tanh,
            name='output'
        )

        self.read = read
        self.output = T.nnet.softmax(layer2.output)
        self.params = read.params + layer1.params + layer2.params
        # self.params = layer1.params + layer2.params
        self.lstm_layers = [layer1]

    def get_initial_states(self):
        return [layer.get_initial_state() for layer in self.lstm_layers]

    def compile(self, updates):
        print("Compiling function")
        hidden_outputs = [layer.output for layer in self.lstm_layers]
        hidden_states = [layer.C for layer in self.lstm_layers]
        self.predict_func = theano.function(inputs=[self.input, self.h_tm1, self.c_tm1],
                                            outputs=[self.output] + hidden_outputs + hidden_states)
        self.train_func = theano.function(
            inputs=[self.input, self.target, self.h_tm1, self.c_tm1],
            outputs=[self.output, self.loss, self.read.g_x, self.read.g_y, self.read.delta, self.read.sigma_sq] + hidden_outputs + hidden_states,
            updates=updates
        )
        print("Done!")

    def train(self, x, y):

        if self.hidden_output is None:
            self.hidden_output = np.zeros((100, 1), dtype=theano.config.floatX)
        if self.hidden_state is None:
            self.hidden_state = np.zeros((100, 1), dtype=theano.config.floatX)

        prediction, loss, px, py, delta, sigma_sq, hidden_output, hidden_state = self.train_func(
            x, y, self.hidden_output, self.hidden_state)
        return prediction, loss, px, py, delta, hidden_state

    def burn_state(self):
        self.hidden_state = None

    def burn_output(self):
        self.hidden_output = None

    def predict(self, x, max_glimpses=4):
        hidden_output = np.zeros((100, 1), dtype=theano.config.floatX)
        hidden_state = np.zeros((100, 1), dtype=theano.config.floatX)
        for i in range(max_glimpses):
            prediction, hidden_output, hidden_state = self.predict_func(
                x, hidden_output, hidden_state)
        return prediction

    def deserialize(self, hidden):
        result = []
        start = 0
        for size in self.lstm_layer_sizes:
            result.append(hidden[start:size].reshape((size, 1)))
            start = start + size
        return result
