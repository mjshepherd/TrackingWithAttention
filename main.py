import theano
import theano.tensor as T
import numpy as np
import models
import matplotlib.pyplot as plt

from load_mnist import load_mnist, pad_mnist, movie_mnist

learning_rate = 0.0001
n_epochs = 10


import pdb


def build_model():

    # instantiate 3D tensor for input
    input = T.matrix(name='input', dtype=theano.config.floatX)
    y = T.vector(name="target", dtype=theano.config.floatX)

    model = models.TestLSTM(input, (100, 100), y)

    NLL = -T.sum(T.log(model.output) * y)
    L2 = sum([T.sum(T.pow(param, 2)) for param in model.params])
    loss = NLL + 0.00001 * L2
    #loss = NLL
    model.set_loss(loss)
    gradients = model.get_grads()
    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(model.params, gradients)
    ]
    model.compile(updates)

    return model


def train_model(model, train_images, train_targets):
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    patience = 1
    improvement_threshold = 1.02

    # Keep track of statistics
    train_error = []

    # Initialize some variables
    error = None
    improvement = 100

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        num_correct = 0
        for i, img in enumerate(train_images):
            # Reset model
            model.burn_output()
            model.burn_state()

            #movie_gen = movie_mnist(img)
            frame, tx, ty = pad_mnist(img)
            frame = frame
            target = train_targets[i]
            old_error = error
            for j in range(20):
                #[frame, tx, ty] = next(movie_gen)
                prediction, loss, px, py, delta, sigma_sq = model.train(frame, target)

            if np.argmax(prediction) == np.argmax(target):
                num_correct += 1

            if i % 1000 == 0 and i > 0:
                percent_correct = num_correct / float(1000)
                print('percent correct: %f' % (percent_correct))
                print(str(px) + ", " + str(py) + ", " + str(delta))
                print("target: " + str(tx) + ", " + str(ty))
                num_correct = 0

            if old_error is not None:
                improvement = error / old_error
            if patience <= epoch and improvement < improvement_threshold:
                print('Breaking due to low improvement')
                done_looping = True
                break

    print('Optimization complete.')

    return train_error

if __name__ == '__main__':
    train_images, train_labels = load_mnist(dataset="testing", path="mnist")

    # convert labels to one-hot encoding
    targets = np.zeros((len(train_labels), 10), dtype=np.uint8)
    for i, label in enumerate(train_labels):
        targets[i, label] = 1
    # make targets a shared vector
    ##targets = theano.shared(targets, dtype=theano.config.floatX)
    model = build_model()

    train_error = train_model(model,
                              train_images,
                              targets)
    pdb.set_trace()
