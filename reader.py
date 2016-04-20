import numpy

import theano
import theano.tensor as T
import pdb


class Reader(object):

    def __init__(self, rng, l, image, image_shape, N, irange=0.001, name='Default_focus'):
        print('Building layer: ' + name)
        # self.input = input
        A = image_shape[0]
        B = image_shape[1]

        mu_ind = T.arange(N, dtype=theano.config.floatX)

        A_ind = T.arange(A, dtype=theano.config.floatX)

        B_ind = T.arange(B, dtype=theano.config.floatX)

        self.g_x = g_x = (A / 2.) * (l[0] + 1)
        self.g_y = g_y = (B / 2.) * (l[1] + 1)
        self.delta = delta = (max(A, B) - 1) / (N - 1) * T.exp(l[2])
        self.sigma_sq = sigma_sq = T.exp(l[3])
        self.gamma = gamma = T.exp(l[4])

        mu_x = g_x + (mu_ind - N / 2. + 0.5) * delta
        mu_y = g_y + (mu_ind - N / 2. + 0.5) * delta

        mu_x_ = mu_x.reshape((N, 1)).repeat(A, axis=1)
        A_ind_ = A_ind.reshape((1, A)).repeat(N, axis=0)
        mu_y_ = mu_y.reshape((N, 1)).repeat(B, axis=1)
        B_ind_ = B_ind.reshape((1, B)).repeat(N, axis=0)
        tol = 1e-4
        # Compute X filter banks##
        F_x = T.exp(-((A_ind_ - mu_x_)**2) / (2 * sigma_sq))
        F_x_div = T.sum(F_x, axis=1)
        F_x = (F_x.T / (F_x_div + tol)).T

        # Compute Y filter banks##
        F_y = T.exp(-((B_ind_ - mu_y_)**2) / (2 * sigma_sq))
        F_y_div = T.sum(F_y, axis=1)
        F_y = (F_y.T / (F_y_div + tol)).T

        self.glimpse = gamma * F_y.dot(image).dot(F_x.T)

if __name__ == "__main__":
    from PIL import Image
    rng = numpy.random.RandomState(23455)
    N = 40
    height = 480
    width = 640
    img = Image.open("cat.jpg")
    img = img.convert('L')
    img = img.resize((640, 480))
    img = numpy.asarray(img)

    l = numpy.asarray([0.417, 0.51, 0.5, -3, 1], dtype=theano.config.floatX)
    l_ = theano.shared(value=l, name='l', borrow=True)
    l_ = l_.reshape((5, 1))

    im_in = T.matrix()
    l_in = T.vector()

    read = Reader(rng,
                  l=l_in,
                  image=im_in,
                  image_shape=(width, height),
                  N=N)

    do_read = theano.function(inputs=[im_in, l_in],
                              outputs=read.glimpse)

    glimpse = do_read(img, l)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(img, cmap='gray')

    plt.figure()
    plt.imshow(glimpse, cmap='gray')

    plt.show()
