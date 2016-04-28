import numpy

import theano
import theano.tensor as T
from attention import ZoomableAttentionWindow
from hiddenlayer import HiddenLayer
from reader import Reader


class ReadLayer(object):

    def __init__(self, rng, h_shape, image_shape, N, name='Default_readlayer'):
        print('Building layer: ' + name)

        self.lin_transform = HiddenLayer(
            rng,
            n_in=h_shape[0] * h_shape[1],
            n_out=4,
            activation=None,
            irange=0.00001,
            name='readlayer: linear transformation')

        # self.zoomable_window = ZoomableAttentionWindow(
        #     channels=1,
        #     img_height=100,
        #     img_width=100,
        #     N=12)
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


if __name__ == "__main__":
    from PIL import Image
    import matplotlib.pyplot as plt
    rng = numpy.random.RandomState(23455)
    N = 100
    height = 480
    width = 640
    learning_rate = 0.000001
    image_shape = (480, 640)
    h_shape = (1, 10)
    img = Image.open("cat.jpg")
    img = img.convert('L')
    img = img.resize((640, 480))
    img = numpy.array(img).reshape((1,) + image_shape)
    img = img / 255.

    target = img[:, 140:240, 280:380]
    target_ = T.tensor3()
    h_ = T.matrix()
    image_ = T.tensor3()

    readlayer = ReadLayer(
        rng,
        h_shape=h_shape,
        image_shape=image_shape,
        N=N)

    read, g_x, g_y, delta, sigma_sq = readlayer.one_step(h_, image_)

    params = readlayer.params

    loss = T.sum((read - target_)**2)
    gradients = T.grad(loss, params)
    updates = updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, gradients)
    ]

    train_func = theano.function(inputs=[h_, image_, target_],
                                 outputs=[
                                     read, loss, g_x, g_y, delta, sigma_sq],
                                 updates=updates,
                                 allow_input_downcast=True)

    h = numpy.random.random(h_shape) * 0
    for i in range(50):
        read, loss, g_x, g_y, delta, sigma_sq = train_func(h, img, target)
        print('Loss: %f, x: %f, y: %f, delta: %f' % (loss, g_x, g_y, delta))

    plt.figure()
    plt.imshow(img[0], cmap='gray')
    plt.figure()
    plt.imshow(target[0], cmap='gray')
    plt.figure()
    plt.imshow(read[0], cmap='gray')
    plt.show()
