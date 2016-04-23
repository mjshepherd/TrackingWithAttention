import numpy

import theano
import theano.tensor as T
import pdb


class Reader(object):

    def __init__(self, rng, image_shape, N, irange=0.001, name='Default_focus'):
        '''
        image_shape: height, width
        '''
        print('Building layer: ' + name)
        # self.input = input
        self.N = N
        A = self.A = image_shape[0]
        B = self.B = image_shape[1]

        self.mu_ind = T.arange(N, dtype=theano.config.floatX)
        self.A_ind = T.arange(A, dtype=theano.config.floatX)
        self.B_ind = T.arange(B, dtype=theano.config.floatX)

    def one_step(self, l, images):
        '''
        l = [n_examples, 5]
        image = [n_examples, height, width]
        '''

        tol = 1e-4
        g_x = (self.B / 2.) * (l[:, 0] + 1)
        g_y = (self.A / 2.) * (l[:, 1] + 1)
        delta = (max(self.A, self.B) - 1) / (self.N - 1) * T.exp(l[:, 2])
        sigma_sq = T.exp(l[:, 3])
        gamma = T.exp(l[:, 4])
        # g_x = g_x.reshape((self.batches, 1))
        # g_y = g_y.reshape((self.batches, 1))
        # delta = delta.reshape((self.batches, 1))

        mu_x = g_x.dimshuffle([0, 'x']) +\
            (self.mu_ind - self.N / 2. + 0.5) * delta.dimshuffle([0, 'x'])
        mu_y = g_y.dimshuffle([0, 'x']) +\
            (self.mu_ind - self.N / 2. + 0.5) * delta.dimshuffle([0, 'x'])

        F_x = T.exp(-((self.B_ind - mu_x.dimshuffle([0, 1, 'x']))**2) / (
            2 * sigma_sq.dimshuffle([0, 'x', 'x'])))
        F_x = F_x / (F_x.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        # Compute Y filter banks##
        F_y = T.exp(-((self.A_ind - mu_y.dimshuffle([0, 1, 'x']))**2) / (
            2 * sigma_sq.dimshuffle([0, 'x', 'x'])))
        # F_y_div = T.sum(F_y, axis=1)
        # F_y = (F_y.T / (F_y_div + tol)).T
        F_y = F_y / (F_y.sum(axis=-1).dimshuffle(0, 1, 'x') + tol)

        read = gamma.dimshuffle([0, 'x', 'x']) * T.batched_dot(T.batched_dot(F_y, images), F_x.dimshuffle([0, 2, 1]))
        return read, g_x, g_y, delta, sigma_sq

if __name__ == "__main__":
    from PIL import Image
    rng = numpy.random.RandomState(23455)
    N = 100
    height = 480
    width = 640
    img = Image.open("cat.jpg")
    img = img.convert('L')
    img = img.resize((640, 480))
    img = numpy.array(img).reshape((1, 480, 640))

    l = numpy.asarray([[0.0, 0.0, -1, 1, 1]], dtype=theano.config.floatX)
    l_ = theano.shared(value=l, name='l', borrow=True)
    l_ = l_.reshape((5, 1))

    im_in = T.tensor3()
    l_in = T.matrix()

    read = Reader(rng,
                  image_shape=(height, width),
                  N=N)
    output, g_x, g_y, delta, sigma_sq = read.one_step(l_in, im_in)

    do_read = theano.function(inputs=[im_in, l_in],
                              outputs=output)

    glimpse = do_read(img, l)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img[0], cmap='gray')

    plt.figure()
    plt.imshow(glimpse[0], cmap='gray')

    plt.show()
