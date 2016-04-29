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

rng = np.random.RandomState(142292)


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

    def __init__(self, input_dims, learning_rate, batch_size):
        self.input = T.tensor3(name='input', dtype=theano.config.floatX)
        self.target = T.matrix(name="target", dtype=theano.config.floatX)
        self.h_tm1 = T.matrix(name="hidden_output", dtype=theano.config.floatX)
        self.c_tm1 = T.matrix(name="hidden_state", dtype=theano.config.floatX)
        self.learning_rate = learning_rate

        N = 12

        self.lstm_layer_sizes = [128, 128]
        self.read_layer = ReadLayer(
            rng,
            h_shape=(reduce(lambda x, y: x + y, self.lstm_layer_sizes), 1),
            image_shape=input_dims,
            N=N,
            name='Read Layer'
        )
        self.conv_layer = ConvPoolLayer(
            rng,
            filter_shape=(30, 1, 3, 3),
            input_shape=(batch_size, 1, N, N),
        )

        self.lstm_layer1 = LSTMLayer(
            rng,
            n_in=N*N,
            n_out=self.lstm_layer_sizes[0],
            name='LSTM1'
        )
        self.lstm_layer2 = LSTMLayer(
            rng,
            n_in=self.lstm_layer_sizes[0],
            n_out=self.lstm_layer_sizes[1],
            name='LSTM2'
        )

        self.output_layer = HiddenLayer(
            rng,
            n_in=self.lstm_layer_sizes[0] + self.lstm_layer_sizes[1] + 5*5*30,
            n_out=10,
            activation=None,
            name='output'
        )

        self.params = self.read_layer.params + self.lstm_layer1.params +\
            self.lstm_layer2.params + self.output_layer.params

    def get_predict_output(self, input, h_tm1, c_tm1):

        h, c, read, g_x, g_y, delta, sigma_sq = self.step_with_att(
            h_tm1, c_tm1, input)
        lin_output = self.output_layer.one_step(h)
        output = T.nnet.softmax(lin_output)
        return output, h, c, read, g_x, g_y, delta, sigma_sq

    def get_train_output(self, images, batch_size):

        images = images.dimshuffle([1, 0, 2, 3])
        h0, c0 = self.get_initial_state(batch_size)
        [h, c, output, g_y, g_x], _ = theano.scan(fn=self.recurrent_step,
                                                  outputs_info=[
                                                      h0, c0, None, None, None],
                                                  sequences=images,
                                                  )
        return output, g_y, g_x

    def recurrent_step(self, image, h_tm1, c_tm1):
        read, g_x, g_y, delta, sigma = self.read_layer.one_step(h_tm1, image)
        
        read_ = read.flatten(ndim=2)

        h_1, c_1 =\
            self.lstm_layer1.one_step(read_,
                                      h_tm1[:, 0:self.lstm_layer_sizes[0]],
                                      c_tm1[:, 0:self.lstm_layer_sizes[0]])
        h_2, c_2 =\
            self.lstm_layer2.one_step(h_1,
                                      h_tm1[:, self.lstm_layer_sizes[0]:],
                                      c_tm1[:, self.lstm_layer_sizes[0]:]
                                      )
        h = T.concatenate([h_1, h_2], axis=1)
        c = T.concatenate([c_1, c_2], axis=1)
        conv = self.conv_layer.one_step(read.dimshuffle([0, 'x', 1, 2]))
        conv = conv.flatten(ndim=2)
        lin_output = self.output_layer.one_step(T.concatenate([h_1, h_2, conv], axis=1))
        output = T.nnet.softmax(lin_output)
        return [h, c, output, g_y, g_x]

    def step_with_att(self, h_tm1, c_tm1, image):
        read, g_x, g_y, delta, sigma_sq = self.read_layer.one_step(
            h_tm1, image)
        read = read.flatten(ndim=2)
        h, c = self.lstm_layer1.one_step(read, h_tm1, c_tm1)

        return [h, c, read, g_x, g_y, delta, sigma_sq]

    def compile(self, train_batch_size):
        print("Compiling functions...")
        train_input = T.tensor4()
        target_y = T.matrix()
        target_x = T.matrix()
        train_output, g_y, g_x = self.get_train_output(train_input,
                                                       train_batch_size)
        classification_loss = self.get_NLL_cost(train_output, self.target)
        tracking_loss = self.get_tracking_cost(g_y, g_x, target_y, target_x)
        loss = 5 * classification_loss + tracking_loss
        updates = Adam(loss, self.params, lr=self.learning_rate)
        # updates = self.get_updates(loss, self.params, self.learning_rate)
        self.train_func = theano.function(
            inputs=[train_input, self.target, target_y, target_x],
            outputs=[train_output[-1], loss],
            updates=updates,
            allow_input_downcast=True
        )

        h_tm1 = T.matrix()
        c_tm1 = T.matrix()
        predict_output, h, c, read, g_x, g_y, delta, sigma_sq = \
            self.get_predict_output(self.input, h_tm1, c_tm1)

        self.predict_func = theano.function(inputs=[self.input, h_tm1, c_tm1],
                                            outputs=[predict_output,
                                                     h,
                                                     c,
                                                     read,
                                                     g_x,
                                                     g_y,
                                                     delta,
                                                     sigma_sq],
                                            allow_input_downcast=True)
        print("Done!")

    def train(self, x, y, target_y, target_x):
        '''
        x is in the form of [batch, time, height, width]
        y is [batch, target]
        '''
        prediction, loss = self.train_func(x, y, target_y, target_x)
        return prediction, loss

    def get_initial_state(self, batch_size, shared=True):
        total_states = reduce(lambda x, y: x + y, self.lstm_layer_sizes)
        h0 = np.zeros((batch_size, total_states), dtype=theano.config.floatX)
        c0 = np.zeros((batch_size, total_states), dtype=theano.config.floatX)
        if shared:
            h0 = theano.shared(
                h0,
                name='h0',
                borrow=True)
            c0 = theano.shared(
                c0,
                name='c0',
                borrow=True)
        return h0, c0
        # initial_state = self.lstm_layer1.initial_hidden_state
        # initial_state = initial_state.dimshuffle(
        #     ['x', 0]).repeat(batch_size, axis=0)
        # return initial_state

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
        NLL = -T.sum((T.log(output) * target), axis=2)
        return NLL.mean()

    def get_tracking_cost(self, g_y, g_x, target_y, target_x):
        loss = (
            (target_y - g_y) ** 2) + ((target_x - g_x) ** 2)
        loss = T.sqrt(loss + 1e-4)
        return loss.mean()

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
