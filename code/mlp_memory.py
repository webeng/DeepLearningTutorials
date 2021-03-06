"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where
instead of feeding the input to the logistic regression you insert a
intermediate layer, called the hidden layer, that has a nonlinear
activation function (usually tanh or sigmoid) . One can use many such
hidden layers making the architecture deep. The tutorial will also tackle
the problem of MNIST digit classification.

.. math::

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5

"""
__docformat__ = 'restructedtext en'


import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T

import cPickle
import random

from logistic_sgd import LogisticRegression, load_data


# start-snippet-1
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
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
        self.input = input
        # end-snippet-1

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
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# start-snippet-2
class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayer.output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input

    def predict(self,x):

        self.input = x

        x = T.matrix('x')

        hl1 = theano.function(
            inputs=[x],
            outputs=T.tanh(T.dot(x, self.hiddenLayer.W) + self.hiddenLayer.b)
        )

        hl1_output = hl1(self.input)

        lr1_output = theano.function(
            inputs=[x],
            outputs=T.nnet.softmax(T.dot(x, self.logRegressionLayer.W) + self.logRegressionLayer.b)
        )

        p_y_given_x = lr1_output(hl1_output)

        y_pred = theano.function(
            inputs = [x],
            outputs = T.argmax(x, axis=1)
        )
        
        predictions = y_pred(p_y_given_x)

        return predictions

def get_learned_functions():

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')

    pkl_file = open( '../data/f1_and_params.pkl', 'rb')
    params = cPickle.load(pkl_file)

    f1_and = MLP(
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=6,
        n_out=2,
    )

    #set up parameters
    f1_and.hiddenLayer.W = params[0]
    f1_and.hiddenLayer.b = params[1]
    f1_and.logRegressionLayer.W = params[2]
    f1_and.logRegressionLayer.b = params[3]

    pkl_file = open( '../data/f1_or_params.pkl', 'rb')
    params = cPickle.load(pkl_file)

    f1_or = MLP(
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=6,
        n_out=2,
    )

    #set up parameters
    f1_or.hiddenLayer.W = params[0]
    f1_or.hiddenLayer.b = params[1]
    f1_or.logRegressionLayer.W = params[2]
    f1_or.logRegressionLayer.b = params[3]
    return f1_and,f1_or


def test_mlp(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=200,
             dataset='mnist.pkl.gz', batch_size=2, n_hidden=6):
    """
    Demonstrate stochastic gradient descent optimization for a multilayer
    perceptron

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
    gradient

    :type L1_reg: float
    :param L1_reg: L1-norm's weight when added to the cost (see
    regularization)

    :type L2_reg: float
    :param L2_reg: L2-norm's weight when added to the cost (see
    regularization)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: the path of the MNIST dataset file from
                 http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz


   """

    f1_and,f1_or = get_learned_functions()

    #datasets = load_data(dataset)

    # test_set_x, test_set_y = datasets[2]
    # print test_set_x.get_value(borrow=True).shape[0]
    # print test_set_y
    datasets = [0,1,2]

    # Go out or not
    X =[
        [0,0,0,0],
        [0,0,0,1],
        [0,0,1,0],
        [0,0,1,1],
        [0,1,0,0],
        [0,1,0,1],
        [0,1,1,0],
        [0,1,1,1],
        [1,0,0,0],
        [1,0,0,1],
        [1,0,1,0],
        [1,0,1,1],
        [1,1,0,0],
        [1,1,0,1],
        [1,1,1,1],
    ]
    # X_reduced = []
    # for x in X:
    #     y_l = f1_or.predict([x[0:2]])[0]
    #     y_r = f1_and.predict([x[2:4]])[0]
    #     X_reduced.append([y_l,y_r])
    
    # X = X_reduced

    Y = [
        1,
        1,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0
    ]

    # X = X * 10
    # Y = Y * 10

    #OR 
    # X = [
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    # ]

    # Y = [
    #     0,
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     1,
    #     1,
    #     0,
    #     1,
    #     1,
    #     1
    # ]

    #AND
    # X = [
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],
    #     [0,0],
    #     [0,1],
    #     [1,0],
    #     [1,1],


    # ]

    # Y = [
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     1,
    #     0,
    #     0,
    #     0,
    #     1,
    # ]


    combined = zip(X, Y)
    random.shuffle(combined)
        
    X[:], Y[:] = zip(*combined)
    # print X
    # print Y
    X = numpy.asarray(X, dtype=theano.config.floatX)
      
    train_length = int(round(len(X) * 0.60))
    valid_length = int(round(len(X) * 0.20))
    test_length = int(round(len(X) * 0.20))

    X_train = X[0:train_length]
    X_valid = X[train_length: (train_length + valid_length)]
    X_test = X[-test_length:]

    Y_train = Y[0:train_length]
    Y_valid = Y[train_length:(train_length + valid_length)]
    Y_test = Y[-test_length:]


    train_set = [X_train,Y_train]
    valid_set = [X_valid,Y_valid]
    test_set = [X_test,Y_test]

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    borrow = True
    train_set_x = theano.shared(numpy.asarray(train_set_x,dtype=theano.config.floatX),borrow=borrow)
    train_set_y = theano.shared(numpy.asarray(train_set_y,dtype=theano.config.floatX), borrow=borrow)

    train_set_y = T.cast(train_set_y, 'int32')

    valid_set_x = theano.shared(numpy.asarray(valid_set_x,dtype=theano.config.floatX),borrow=borrow)
    valid_set_y = theano.shared(numpy.asarray(valid_set_y,dtype=theano.config.floatX),borrow=borrow)

    valid_set_y = T.cast(valid_set_y, 'int32')

    test_set_x = theano.shared(numpy.asarray(test_set_x,dtype=theano.config.floatX),borrow=borrow)
    test_set_y = theano.shared(numpy.asarray(test_set_y,dtype=theano.config.floatX),borrow=borrow)
    
    test_set_y = T.cast(test_set_y, 'int32')
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=n_hidden,
        n_out=2
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )
    # end-snippet-4

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    # end-snippet-5

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

            # update train set X train_set_x[index * batch_size: (index + 1) * batch_size],
            #get current batch
            x_minibatch = train_set_x.get_value()
            y_minibatch = train_set_x.get_value()

            x_minibatch = x_minibatch[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
            y_minibatch = y_minibatch[minibatch_index * batch_size: (minibatch_index + 1) * batch_size][0]

            X_reduced = []
            Y_reduced = []

            index = 0
            for x in x_minibatch:
                
                print "x"
                print x

                y_l = f1_or.predict([x[0:2]])[0]
                y_r = f1_or.predict([x[2:4]])[0]

                X_reduced.append([y_l,y_r])
                Y_reduced.append(y_minibatch[index])

                y_l = f1_or.predict([x[0:2]])[0]
                y_r = f1_and.predict([x[2:4]])[0]

                X_reduced.append([y_l,y_r])
                Y_reduced.append(y_minibatch[index])

                y_l = f1_and.predict([x[0:2]])[0]
                y_r = f1_or.predict([x[2:4]])[0]

                X_reduced.append([y_l,y_r])
                Y_reduced.append(y_minibatch[index])

                y_l = f1_and.predict([x[0:2]])[0]
                y_r = f1_and.predict([x[2:4]])[0]

                X_reduced.append([y_l,y_r])
                Y_reduced.append(y_minibatch[index])

                print "current prediction"
                #print classifier.predict([X_reduced[0]])
                print classifier.predict(X_reduced)

                print "X_reduced"
                print X_reduced

                print "Y_reduced"
                print Y_reduced
                # calculate best option
                index += 1
                break

            #print x_minibatch

            minibatch_avg_cost = train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    
"""    #classifier.input = datasets[0][0]
    #print X[0]
    #
    #classifier.input = X[0:4]

    #print classifier.predict(X[0:4])

    # minimum hidden units to learn or is 2 and to learn AND is 6
    params = classifier.params

    # save or retrieve saved params
    output = open('../data/f1_and_params.pkl', 'wb')
    cPickle.dump(params, output,protocol=-1)
    output.close()

    pkl_file = open( '../data/f1_and_params.pkl', 'rb')
    params = cPickle.load(pkl_file)

    f1_or = MLP(
        rng=rng,
        input=x,
        n_in=2,
        n_hidden=n_hidden,
        n_out=2,
    )

    #set up parameters
    f1_or.hiddenLayer.W = params[0]
    f1_or.hiddenLayer.b = params[1]
    f1_or.logRegressionLayer.W = params[2]
    f1_or.logRegressionLayer.b = params[3]

    #print f1_or.predict(datasets[0][0])
    print X[0:2]
    print f1_or.predict(X[0:2])"""

    # print numpy.tanh(numpy.dot(classifier.input,params[0].get_value()) + params[1].get_value())
    # print "----------------------------"

    #print classifier_or.hiddenLayer.W.get_value()
    #print theano.printing.pydotprint(predict_or,'../data/predict_or_graph.jpg')
    #print theano.printing.debugprint(predict_or)
    

if __name__ == '__main__':
    test_mlp()
