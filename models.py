import numpy as np
import theano
import theano.tensor as T
from theano_lstm import *
from hiddenlayer import HiddenLayer
from convpoollayer import ConvPoolLayer
from LSTMLayer import LSTMLayer
from readlayer import ReadLayer
from adam import Adam


import pdb

rng = np.random.RandomState(23455)


class AbstractModel():
    def get_grads(self, loss):
        return T.grad(loss, self.params)

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

    def __init__(self, input_dims, learning_rate):
        self.input = T.tensor3(name='input', dtype=theano.config.floatX)
        self.target = T.matrix(name="target", dtype=theano.config.floatX)
        self.h_tm1 = T.matrix(name="hidden_output", dtype=theano.config.floatX)
        self.c_tm1 = T.matrix(name="hidden_state", dtype=theano.config.floatX)
        self.learning_rate = learning_rate

        self.lstm_layer_sizes = [256]
        self.read_layer = ReadLayer(
            rng,
            h_shape=(self.lstm_layer_sizes[0], 1),
            image_shape=input_dims,
            N=12,
            name='Read Layer'
        )

        self.f_g = HiddenLayer(
            rng,
            n_in=12 * 12 + self.lstm_layer_sizes[0] + 4,
            n_out=500,
            activation=T.nnet.sigmoid,
            name='gangsta_func')

        self.lstm_layer1 = LSTM(
            input_size=500,
            hidden_size=self.lstm_layer_sizes[0],
            activation=T.tanh,
            clip_gradients=False)

        # self.lstm_layer1 = LSTMLayer(
        #     rng,
        #     n_in=144,
        #     n_out=self.lstm_layer_sizes[0],
        #     name='LSTM1'
        # )

        self.output_layer = HiddenLayer(
            rng,
            n_in=self.lstm_layer_sizes[0],
            n_out=10,
            activation=None,
            name='output'
        )

        self.params = self.read_layer.params + self.lstm_layer1.params +\
            self.output_layer.params + self.f_g.params
        self.lstm_layers = [self.lstm_layer1]

    def get_predict_output(self, input, h_tm1, c_tm1):
        in_state = T.concatenate([c_tm1, h_tm1], axis=1)
        out_state, read, g_x, g_y, delta, sigma_sq = self.step_with_att(
            in_state, input)
        h = self.lstm_layer1.postprocess_activation(out_state)
        lin_output = self.output_layer.one_step(h)
        output = T.nnet.softmax(lin_output)
        return output, h, out_state, read, g_x, g_y, delta, sigma_sq

    def get_train_output(self, images, batch_size):

        images = images.dimshuffle([1, 0, 2, 3])
        initial_state = self.get_initial_state(batch_size)
        state, _ = theano.scan(fn=self.recurrent_step,
                               outputs_info=initial_state,
                               sequences=images,
                               )
        final_state = state[-1]
        h = self.lstm_layer1.postprocess_activation(final_state)
        lin_output = self.output_layer.one_step(h)
        return T.nnet.softmax(lin_output)

    def recurrent_step(self, image, state_tm1):
        h_tm1 = self.lstm_layer1.postprocess_activation(state_tm1)
        read, g_x, g_y, delta, sigma = self.read_layer.one_step(h_tm1, image)
        read = read.flatten(ndim=2)
        hidden_rep = self.f_g.one_step(T.concatenate([read, h_tm1, g_x.dimshuffle([0, 'x']),
                                                      g_y.dimshuffle([0, 'x']),
                                                      delta.dimshuffle(
                                                          [0, 'x']),
                                                      sigma.dimshuffle([0, 'x'])], axis=1))
        # h, c = self.lstm_layer1.one_step(read, h_tm1, c_tm1)
        lstm_out = self.lstm_layer1.activate(hidden_rep, state_tm1)
        return [lstm_out]

    def step_with_att(self, state_tm1, image):
        h_tm1 = self.lstm_layer1.postprocess_activation(state_tm1)
        read, g_x, g_y, delta, sigma_sq = self.read_layer.one_step(
            h_tm1, image)
        read = read.flatten(ndim=2)
        #h, c = self.lstm_layer1.one_step(read, h_tm1, c_tm1)
        lstm_out = self.lstm_layer1.activate(read, state_tm1)

        return [lstm_out, read, g_x, g_y, delta, sigma_sq]

    def compile(self, train_batch_size):
        print("Compiling functions...")
        train_input = T.tensor4()
        train_output = self.get_train_output(train_input,
                                             train_batch_size)
        loss = self.get_NLL_cost(train_output, self.target)
        updates = Adam(loss, self.params, lr=self.learning_rate)
        #updates = self.get_updates(loss, self.params, self.learning_rate)
        self.train_func = theano.function(
            inputs=[train_input, self.target],
            outputs=[train_output, loss],
            updates=updates,
            allow_input_downcast=True
        )

        h_tm1 = T.matrix()
        c_tm1 = T.matrix()
        predict_output, h, lstm_out, read, g_x, g_y, delta, sigma_sq = \
            self.get_predict_output(self.input, h_tm1, c_tm1)

        self.predict_func = theano.function(inputs=[self.input, h_tm1, c_tm1],
                                            outputs=[predict_output,
                                                     h,
                                                     lstm_out,
                                                     read,
                                                     g_x,
                                                     g_y,
                                                     delta,
                                                     sigma_sq],
                                            allow_input_downcast=True)
        print("Done!")

    def train(self, x, y):
        '''
        x is in the form of [batch, time, height, width]
        y is [batch, target]
        '''
        prediction, loss = self.train_func(x, y)
        return prediction, loss

    def get_initial_state(self, batch_size, shared=True):
        # total_states = reduce(lambda x, y: x + y, self.lstm_layer_sizes)
        # h0 = np.zeros((batch_size, total_states), dtype=theano.config.floatX)
        # c0 = np.zeros((batch_size, total_states), dtype=theano.config.floatX)
        # if shared:
        #     h0 = theano.shared(
        #         h0,
        #         name='h0',
        #         borrow=True)
        #     c0 = theano.shared(
        #         c0,
        #         name='c0',
        #         borrow=True)
        # return h0, c0
        initial_state = self.lstm_layer1.initial_hidden_state
        initial_state = initial_state.dimshuffle(
            ['x', 0]).repeat(batch_size, axis=0)
        return initial_state

    def predict(self, x, reset=True):
        if reset:
            self.predict_h, self.predict_c = self.get_initial_state(
                1, shared=False)

        if len(x.shape) == 2:
            x = np.expand_dims(x, axis=0)

        prediction, self.predict_h, self.predict_c, read, g_x, g_y, delta, sigma_sq =\
            self.predict_func(x, self.predict_h, self.predict_c)

        return prediction, [read, g_x, g_y, delta, sigma_sq]

    def get_NLL_cost(self, output, target):
        NLL = -T.sum((T.log(output) * target), axis=1)
        return NLL.mean()

    def get_updates(self, cost, params, learning_rate):
        gradients = T.grad(cost, params)
        updates = updates = [
            (param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, gradients)
        ]
        return updates

    def deserialize(self, hidden):
        result = []
        start = 0
        for size in self.lstm_layer_sizes:
            result.append(hidden[start:size].reshape((size, 1)))
            start = start + size
        return result
