"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import timeit

import numpy
import cPickle

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd_test import LogisticRegression, load_data
from mlp import HiddenLayer
from fetex_image import FetexImage


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input


# def evaluate_lenet5(learning_rate=0.1, n_epochs=100,
#                     dataset='mnist.pkl.gz',
#                     nkerns=[(96) , (256)], batch_size=300):
def evaluate_lenet5(learning_rate=0.1, n_epochs=2,
                    dataset='mnist.pkl.gz',
                    nkerns=[(25 / 1) , (25 / 1)], batch_size=400):

    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    #print train_set_x
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    #index = 0
    #print test_set_x[index * batch_size: (index + 1) * batch_size]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    #y = T.lvector('y')  # the labels are presented as 1D vector of
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels


    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # :param filter_shape: (number of filters, num input feature maps,
    #                          filter height, filter width)
    
    #:type image_shape: tuple or list of length 4
    #    :param image_shape: (batch size, num input feature maps,
    #                         image height, image width)

    # Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (28, 28) is the size of MNIST images.
    #layer0_input = x.reshape((batch_size, 1, 28, 28))
    #layer0_input = x.reshape((batch_size, 1, 256, 256))
    #layer0_input = x.reshape((batch_size, 3, 256, 256))
    layer0_input = x.reshape((batch_size, 3, 64, 64))
    #layer0_input = x

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (64-5+1 , 64-5+1) = (60, 60)
    # maxpooling reduces this further to (60/2, 60/2) = (30, 30)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 30, 30)

    layer0 = LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 3, 64, 64),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (30-5+1, 30-5+1) = (26, 26)
    # maxpooling reduces this further to (26/2, 26/2) = (18, 18)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 13, 13)
    layer1 = LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 30, 30),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4),
    # or (500, 50 * 4 * 4) = (500, 800) with the default values.
    layer2_input = layer1.output.flatten(2)
    #layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = HiddenLayer(
        rng,
        input=layer2_input,
        n_in=nkerns[1] * 13 * 13,
        n_out=400,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=10)
    #layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=2)
    layer4 = LogisticRegression(input=layer3.output, n_in=400, n_out=21)

    # the cost we minimize during training is the NLL of the model
    #cost = layer3.negative_log_likelihood(y)
    cost = layer4.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer4.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    #params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    #params = layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    params = layer4.params + layer3.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-1

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            print iter
            if iter % 100 == 0:
                print 'training @ iter = ', iter
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

                    output = open('../data/layer0.pkl', 'wb')
                    cPickle.dump(layer0, output,protocol=-1)
                    output.close()

                    output = open('../data/layer1.pkl', 'wb')
                    cPickle.dump(layer1, output,protocol=-1)
                    output.close()

                    # output = open('../data/layer2.pkl', 'wb')
                    # cPickle.dump(layer2, output,protocol=-1)
                    # output.close()

                    output = open('../data/layer3.pkl', 'wb')
                    cPickle.dump(layer3, output,protocol=-1)
                    output.close()

                    output = open('../data/layer4.pkl', 'wb')
                    cPickle.dump(layer4, output,protocol=-1)
                    output.close()
                    
                    # save the best model
                    # with open('best_model.pkl', 'w') as f:
                    #     cPickle.dump(layer0, f)                                                
                    #     cPickle.dump(layer1, f)                        
                    #     cPickle.dump(layer2, f)                        
                    #     cPickle.dump(layer3, f)
                    #     cPickle.dump(layer4, f)

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))

# def predict(self, data):
#     """
#     the CNN expects inputs with Nsamples = self.batch_size.
#     In order to run 'predict' on an arbitrary number of samples we
#     pad as necessary.
#     """
#     if isinstance(data, list):
#         data = np.array(data)
#     if data.ndim == 1:
#         data = np.array([data])

#     nsamples = data.shape[0]
#     n_batches = nsamples//self.batch_size
#     n_rem = nsamples%self.batch_size
#     if n_batches > 0:
#         preds = [list(self.predict_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
#                                        for i in range(n_batches)]
#     else:
#         preds = []
#     if n_rem > 0:
#         z = np.zeros((self.batch_size, self.n_in * self.n_in))
#         z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
#         preds.append(self.predict_wrap(z)[0:n_rem])
    
#     return np.hstack(preds).flatten()

def cosine_distance(a, b):
    import numpy as np
    from numpy import linalg as LA 
    dot_product =  np.dot(a,b.T)
    cosine_distance = dot_product / (LA.norm(a) * LA.norm(b))
    return cosine_distance

def predict():

    from sktheano_cnn import MetaCNN as CNN
    cnn = CNN()

    pkl_file = open( '../data/train_set.pkl', 'rb')
    train_set = cPickle.load(pkl_file)

    pkl_file = open( '../data/valid_set.pkl', 'rb')
    valid_set = cPickle.load(pkl_file)

    pkl_file = open( '../data/test_set.pkl', 'rb')
    test_set = cPickle.load(pkl_file)

    """An example of how to load a trained model and use it
    to predict labels.
    """

    fe = FetexImage(verbose=True)
    # load the saved model
    classifier = cPickle.load(open('best_model.pkl'))

    layer0 = cPickle.load(open('../data/layer0.pkl'))
    layer1 = cPickle.load(open('../data/layer1.pkl'))
    # layer2 = cPickle.load(open('../data/layer2.pkl')) 
    layer3 = cPickle.load(open('../data/layer3.pkl'))
    layer4 = cPickle.load(open('../data/layer4.pkl'))

    #layer0_input = x.reshape((batch_size, 3, 64, 64))

    # predict = theano.function(
    #     outputs=layer4.y_pred,
    #     givens = {x : train_set_x[0] }
    # )

    # compile a predictor function
    predict_model = theano.function(
        inputs=[classifier.input],
        outputs=classifier.y_pred)

    # We can test it on some examples from test test
    dataset='mnist.pkl.gz'
    datasets = load_data(dataset)
    
    test_set_x, test_set_y = datasets[2]
    test_set_x = test_set_x.get_value()
    
    train_set_x, train_set_y = datasets[0]
    train_set_x = train_set_x.get_value()

    pkl_file = open( '../data/X_original.pkl', 'rb')
    X_original = cPickle.load(pkl_file)

    a = X_original[0]
    #fe.reconstructImage(a).show()

    #predicted_values = predict_model([a])

    get_input = theano.function(
        inputs=[classifier.input],
        outputs=classifier.input
    )
    
    a = get_input(train_set_x[0:1])
    #print a.shape
    
    x = T.matrix('x')   # the data is presented as rasterized images
    predict = theano.function(
        inputs = [x],
        outputs=layer3.output
    )
    # givens = { x : train_set_x[0] }
    #train_set_x = train_set_x[0:400]
    #x = train_set_x.reshape((400, 3, 64, 64))
    x = train_set_x.reshape(np.zeros((400,3,64,64)))
    print predict(x)
    #predicted_values = predict_model([train_set_x[0]])
    #print predicted_values
    return "fffff"


    max_similarity = 0
    max_similarity_pos = -1
    #for i in xrange(1,len(train_set_x)):
    for i in xrange(1,1000):
        b = get_input([train_set_x[i]])
        d = cosine_distance(a, b)
        if d > max_similarity:
            max_similarity = d
            max_similarity_pos = i

    fe.reconstructImage(X_original[max_similarity_pos]).show()

    #a = a.flatten(order='F')
    # a = a * 256
    # a = numpy.array(a,dtype=numpy.uint8)

    #b = b.flatten(order='F')
    # b = b * 256
    # b = numpy.array(b,dtype=numpy.uint8)

    # a = get_input([a])
    # b = get_input([b])

    print a.shape
    print b.shape
    print cosine_distance(a, b)

    # #print get_input(test_set_x[0:1]).sum()

    # print ("Predicted values for the first 10 examples in test set:")
    # print predicted_values

if __name__ == '__main__':
    #evaluate_lenet5()
    predict()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
