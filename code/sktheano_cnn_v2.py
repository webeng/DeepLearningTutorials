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

"""
Aaron Berndsen:
A Conformal Neural Network using Theano for computation and structure, 
but built to obey sklearn's basic 'fit' 'predict' functionality

*code largely motivated from deeplearning.net examples
and Graham Taylor's "Vanilla RNN" (https://github.com/gwtaylor/theano-rnn/blob/master/rnn.py)

You'll require theano and libblas-dev

tips/tricks/notes:
* if training set is large (>O(100)) and redundant, use stochastic gradient descent (batch_size=1), otherwise use conjugate descent (batch_size > 1)
*  

Basic usage:
import nnetwork as NN

n = NN.NeuralNetwork(design=[8,8]) # a NN with two hidden layers of 8 neurons each
n.fit(Xtrain, ytrain)
pred = n.predict(Xtest)

"""
import cPickle as pickle
import logging
import numpy as np
import timeit

from sklearn.base import BaseEstimator
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import logging
import os
import sys
from logistic_sgd_test import LogisticRegression, load_data
_logger = logging.getLogger("theano.gof.compilelock")
_logger.setLevel(logging.WARN)
logger = logging.getLogger(__name__)

root = logging.getLogger()
root.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(message)s')
ch.setFormatter(formatter)
root.addHandler(ch)

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'

class CNN(object):
	"""
	Convolutional Neural Network (CNN), 
	backend by Theano, but compliant with sklearn interface.

	This class defines all the layers in the network. At present the CNN has 7 layers: 3 LeNetConvPoolLayer, 
	3 MLP HiddenLayers and 1 LogisticRegression. This architecture is for classifying 128x128 grayscale images.
	The class MetaCNN has more lower level routines such as initialization, prediction and save.

	You should init with MetaCNN.
	"""

	def __init__(self, input, im_width=128, im_height=128, n_out=2, activation=T.tanh,
				 nkerns=[48,128,256],
				 filters=[13,5,4],
				 poolsize=[(2,2),(2,2),(2,2)],
				 n_hidden=[200,50,2],
				 output_type='softmax', batch_size=128,
				 use_symbolic_softmax=False,verbose = True):

		"""
		im_width : width of input image
		im_height : height of input image
		n_out : number of class labels
		
		:type nkerns: list of integers
		:param nkerns: number of kernels on each layer
		
		:type filters: list of integers
		:param filters: width of convolution

		:type poolsize: list of 2-tuples
		:param poolsize: maxpooling in convolution layer (index-0),
						 and direction x or y (index-1)

		:type n_hidden: list of integers
		:param n_hidden: number of hidden neurons
		
		:type output_type: string
		:param output_type: type of decision 'softmax', 'binary', 'real'
		
		:type batch_size: integers
		:param batch_size: number of samples in each training batch. Default 200.
		"""    

		self.activation = activation
		self.output_type = output_type
		self.verbose = verbose

		if verbose:
			logger.info("\n Input image with:{} height:{} ".format(im_width,im_height))

		# if use_symbolic_softmax:
		#     def symbolic_softmax(x):
		#         e = T.exp(x)
		#         return e / T.sum(e, axis=1).dimshuffle(0, 'x')
		#     self.softmax = symbolic_softmax
		# else:
		#     self.softmax = T.nnet.softmax

		rng = np.random.RandomState(23455)

		# Reshape matrix of rasterized images of shape (batch_size, nx*ny)
		# to a 4D tensor, compatible with our LeNetConvPoolLayer
		layer0_input = input.reshape((batch_size, 1, im_width, im_height))

		# Construct the first convolutional pooling layer:
		# filtering reduces the image size to (im_width - filters[0]+1, im_height-filters[0] + 1 )=(x,x)
		# maxpooling reduces this further to (x/2,x/2) = (y,y)
		# 4D output tensor is thus of shape (batch_size,nkerns[0],y,y)
		self.layer0 = LeNetConvPoolLayer(
			rng, 
			input=layer0_input,
			image_shape=(batch_size, 1, im_width, im_height),
			filter_shape=(nkerns[0], 1, filters[0], filters[0]),
			poolsize=poolsize[0]
		)

		if self.verbose:
			logger.info('\n Layer {} \n image_shape: ({},{},{},{}) \n filter_shape: ({},{},{},{}) \n poolsize:{}'.format(0,
				batch_size, 1, im_width, im_height, 
				nkerns[0], 1, filters[0], filters[0],
				poolsize[0])
			)
		
		# Construct the second convolutional pooling layer
		# filtering reduces the image size to (im_width-filters[0]+1,im_height-filters[0]+1) = (x,x)
		# maxpooling reduces this further to (x/2,x/2) = y
		# 4D output tensor is thus of shape (nkerns[0],nkerns[1],y,y)
		im_width_l1 = (im_width - filters[0] + 1)/poolsize[0][0]
		im_height_l1 = (im_height - filters[0] + 1)/poolsize[0][1]
		
		self.layer1 = LeNetConvPoolLayer(
			rng,
			input=self.layer0.output,
			image_shape=(batch_size, nkerns[0], im_width_l1, im_height_l1),
			filter_shape=(nkerns[1], nkerns[0], filters[1], filters[1]),
			poolsize=poolsize[1]
		)

		if self.verbose:
			logger.info('\n Layer {} \n image_shape: ({},{},{},{}) \n filter_shape: ({},{},{},{}) \n poolsize:{}'.format(1
				,batch_size, nkerns[0], im_width_l1, im_height_l1, 
				nkerns[1], nkerns[0], filters[1], filters[1],
				poolsize[1])
			)

		# Construct the third convolutional pooling layer
		# filtering reduces the image size to (im_width_l1-filters[1]+1,im_height_l1-filters[1]+1) = (x,x)
		# maxpooling reduces this further to (x/2,x/2) = y
		# 4D output tensor is thus of shape (nkerns[1],nkerns[2],y,y)
		im_width_l2 = (im_width_l1 - filters[1] + 1)/poolsize[1][0]
		im_height_l2 = (im_height_l1 - filters[1] + 1)/poolsize[1][1]

		self.layer2 = LeNetConvPoolLayer(
			rng,
			input=self.layer1.output,
			image_shape=(batch_size, nkerns[1], im_width_l2, im_height_l2),
			filter_shape=(nkerns[2], nkerns[1], filters[2], filters[2]),
			poolsize=poolsize[2]
		)

		if self.verbose:
			logger.info('\n Layer {} \n image_shape: ({},{},{},{}) \n filter_shape: ({},{},{},{}) \n poolsize:{}'.format(2,
				batch_size, nkerns[1], im_width_l2, im_height_l2, 
				nkerns[2], nkerns[1], filters[2], filters[2],
				poolsize[2])
			)

		# the TanhLayer being fully-connected, it operates on 2D matrices of
		# shape (batch_size,num_pixels) (i.e matrix of rasterized images).
		# This will generate a matrix of shape (20,32*4*4) = (20,512)
		layer3_input = self.layer2.output.flatten(2)

	   # construct a fully-connected sigmoidal layer
		im_width_l3 = (im_width_l2-filters[2]+1)/poolsize[2][0]
		im_height_l3 = (im_height_l2-filters[2]+1)/poolsize[2][1]

		self.layer3 = HiddenLayer(
			rng,
			input=layer3_input,
			n_in=nkerns[2] * im_width_l3 * im_height_l3,
			n_out=n_hidden[0], 
			activation=T.tanh
		)

		if self.verbose:
			logger.info("\n Layer {} input: ({},{})".format(3,batch_size,nkerns[2] * im_width_l3 * im_height_l3))
		
		# construct a fully-connected sigmoidal layer
		self.layer4 = HiddenLayer(
			rng,
			input=self.layer3.output,
			n_in=n_hidden[0],
			n_out=n_hidden[1], 
			activation=T.tanh
		)

		if self.verbose:
			logger.info("\n Layer {} input: {}".format(4,n_hidden[1]))
		
		# construct a fully-connected sigmoidal layer
		self.layer5 = HiddenLayer(
			rng,
			input=self.layer4.output,
			n_in=n_hidden[1],
			n_out=n_hidden[2], 
			activation=T.tanh
		)

		if self.verbose:
			logger.info("\n Layer {} input: {}".format(5,n_hidden[2]))

		# classify the values of the fully-connected sigmoidal layer
		self.layer6 = LogisticRegression(
			input=self.layer5.output,
			n_in=n_hidden[2],
			n_out=n_out
		)

		if self.verbose:
			logger.info("\n Layer {} input: {}".format(6,n_hidden[2]))

		# CNN regularization
		self.L1 = self.layer6.L1
		self.L2_sqr = self.layer6.L2_sqr
		
		# create a list of all model parameters to be fit by gradient descent
		self.params = self.layer6.params + self.layer5.params + self.layer4.params + self.layer3.params + self.layer2.params + self.layer1.params + self.layer0.params

		self.y_pred = self.layer6.y_pred
		self.p_y_given_x = self.layer6.p_y_given_x
		#self.layer3_output = self.layer5.input
		self.layer5_output = self.layer5.input

		if self.output_type == 'real':
			self.loss = lambda y: self.mse(y)
		elif self.output_type == 'binary':
			self.loss = lambda y: self.nll_binary(y)
		elif self.output_type == 'softmax':
			# push through softmax, computing vector of class-membership
			# probabilities in symbolic form
			self.loss = lambda y: self.nll_multiclass(y)
		else:
			raise NotImplementedError

	def mse(self, y):
		# error between output and target
		return T.mean((self.y_pred - y) ** 2)

	def nll_binary(self, y):
		# negative log likelihood based on binary cross entropy error
		return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

	#same as negative-log-likelikhood
	def nll_multiclass(self, y):
		# negative log likelihood based on multiclass cross entropy error
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of time steps (call it T) in the sequence
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""Return a float representing the number of errors in the sequence
		over the total number of examples in the sequence ; zero one
		loss over the size of the sequence

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""
		# check if y has same dimension of y_pred
		if y.ndim != self.y_out.ndim:
			raise TypeError('y should have the same shape as self.y_out',
				('y', y.type, 'y_pred', self.y_pred.type))

		if self.output_type in ('binary', 'softmax'):
			# check if y is of the correct datatype
			if y.dtype.startswith('int'):
				# the T.neq operator returns a vector of 0s and 1s, where 1
				# represents a mistake in prediction
				return T.mean(T.neq(self.y_pred, y))
			else:
				raise NotImplementedError()

class MetaCNN(BaseEstimator):
	"""
	the actual CNN is not init-ed until .fit is called.
	We determine the image input size (assumed square images) and
	the number of outputs in .fit from the training data
	"""
	def __init__(
		self, learning_rate=0.05, n_epochs=60, batch_size=128, activation='tanh', 
	   nkerns=[20,45], n_hidden=500, filters=[15,7], poolsize=[(3,3),(2,2)],
	   output_type='softmax',L1_reg=0.00, L2_reg=0.00,
	   use_symbolic_softmax=False, im_width=128, im_height=128, n_out=2,verbose = True):

		self.learning_rate = float(learning_rate)
		self.nkerns = nkerns
		self.n_hidden = n_hidden
		self.filters = filters
		self.poolsize = poolsize
		self.n_epochs = int(n_epochs)
		self.batch_size = int(batch_size)
		self.L1_reg = float(L1_reg)
		self.L2_reg = float(L2_reg)
		self.activation = activation
		self.output_type = output_type
		self.use_symbolic_softmax = use_symbolic_softmax
		self.im_width = im_width
		self.im_height = im_height
		self.n_out = n_out
		self.verbose = verbose

	def ready(self):
		"""
		this routine is called from "fit" since we determine the
		image size (assumed square) and output labels from the training data.
		"""

		#input
		self.x = T.matrix('x')
		#output (a label)
		self.y = T.ivector('y')
		
		if self.activation == 'tanh':
			activation = T.tanh
		elif self.activation == 'sigmoid':
			activation = T.nnet.sigmoid
		elif self.activation == 'relu':
			activation = lambda x: x * (x > 0)
		elif self.activation == 'cappedrelu':
			activation = lambda x: T.minimum(x * (x > 0), 6)
		else:
			raise NotImplementedError
		
		self.cnn = CNN(
			input=self.x, 
			n_out=self.n_out, 
			activation=activation, 
			nkerns=self.nkerns,
			filters=self.filters,
			n_hidden=self.n_hidden,
			poolsize=self.poolsize,
			output_type=self.output_type,
			batch_size=self.batch_size,
			use_symbolic_softmax=self.use_symbolic_softmax,
			verbose=self.verbose
		)
		
		#self.cnn.predict expects batch_size number of inputs. 
		#we wrap those functions and pad as necessary in 'def predict' and 'def predict_proba'
		self.predict_wrap = theano.function(inputs=[self.x],
											outputs=self.cnn.y_pred,
											mode=mode)

		# self.predict_vector = theano.function(inputs=[self.x],
		# 									outputs=self.cnn.layer5.output,
		# 									mode=mode)
		self.predict_vector = theano.function(inputs=[self.x],
											outputs=self.cnn.layer5_output,
											mode=mode)
		self.predict_proba_wrap = theano.function(inputs=[self.x],
												  outputs=self.cnn.p_y_given_x,
												  mode=mode)

	def score(self, X, y):
		"""Returns the mean accuracy on the given test data and labels.

		Parameters
		----------
		X : array-like, shape = [n_samples, n_features]
			Training set.

		y : array-like, shape = [n_samples]
			Labels for X.

		Returns
		-------
		z : float

		"""
		return np.mean(self.predict(X) == y)

	def fit(self, train_set_x, train_set_y, valid_set_x=None, valid_set_y=None,test_set_x = None,test_set_y = None,
			n_epochs=None):
		""" Fit model

		Pass in X_test, Y_test to compute test error and report during
		training.

		X_train : ndarray (T x n_in)
		Y_train : ndarray (T x n_out)

		validation_frequency : int
			in terms of number of sequences (or number of weight updates)
		n_epochs : None (used to override self.n_epochs from init.
		"""

		self.ready()

		# compute number of minibatches for training, validation and testing
		n_train_batches = train_set_x.get_value(borrow=True).shape[0]
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_test_batches = test_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= self.batch_size
		n_valid_batches /= self.batch_size
		n_test_batches /= self.batch_size

		######################
		# BUILD ACTUAL MODEL #
		######################
		if self.verbose:
			logger.info('\n ... building the model')

		index = T.lscalar('index')    # index to a [mini]batch

		# cost = self.cnn.loss(self.y)\
		#     + self.L1_reg * self.cnn.L1\
		#     + self.L2_reg * self.cnn.L2_sqr
		#cost = self.cnn.loss(self.y)
		cost = self.cnn.layer6.negative_log_likelihood(self.y)
		#self.cnn.loss(self.y),
		test_model = theano.function(
			[index],
			self.cnn.layer6.errors(self.y),
			givens={
				self.x: test_set_x[index * self.batch_size: (index + 1) * self.batch_size],
				self.y: test_set_y[index * self.batch_size: (index + 1) * self.batch_size]
			}
		)
		#self.cnn.loss(self.y),
		validate_model = theano.function(
			[index],
			self.cnn.layer6.errors(self.y),
			givens={
				self.x: valid_set_x[index * self.batch_size: (index + 1) * self.batch_size],
				self.y: valid_set_y[index * self.batch_size: (index + 1) * self.batch_size]
			}
		)

		# create a list of all model parameters to be fit by gradient descent
		self.params = self.cnn.params

		# create a list of gradients for all model parameters
		self.grads = T.grad(cost, self.params)
		
		# train_model is a function that updates the model parameters by
		# SGD Since this model has many parameters, it would be tedious to
		# manually create an update rule for each model parameter. We thus
		# create the updates dictionary by automatically looping over all
		# (params[i],grads[i]) pairs.
		# self.updates = {}
		# for param_i, grad_i in zip(self.params, self.grads):
		#     self.updates[param_i] = param_i - self.learning_rate * grad_i

		self.updates = [
		(param_i, param_i - self.learning_rate * grad_i)
		for param_i, grad_i in zip(self.params, self.grads)
		]

		train_model = theano.function(
			[index], 
			cost, 
			updates=self.updates,
			givens={
				self.x: train_set_x[index * self.batch_size: (index + 1) * self.batch_size],
				self.y: train_set_y[index * self.batch_size: (index + 1) * self.batch_size]
			}
		)

		###############
		# TRAIN MODEL #
		###############
		if self.verbose:
			logger.info('\n... training')

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

		best_validation_loss = np.inf
		best_iter = 0
		test_score = 0.
		start_time = timeit.default_timer()

		epoch = 0
		done_looping = False

		while (epoch < n_epochs) and (not done_looping):
			epoch = epoch + 1
			for minibatch_index in xrange(n_train_batches):
				iter = (epoch - 1) * n_train_batches + minibatch_index

				if iter % 100 == 0:
					logger.info('... training @ iter = {}'.format(iter))
				cost_ij = train_model(minibatch_index)
				print cost_ij
				if (iter + 1) % validation_frequency == 0:

					# compute zero-one loss on validation set
					validation_losses = [validate_model(i) for i
										 in xrange(n_valid_batches)]
					this_validation_loss = np.mean(validation_losses)
					logger.info('epoch %i, minibatch %i/%i, validation error %f %%' %
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
						test_score = np.mean(test_losses)
						logger.info(('     epoch %i, minibatch %i/%i, test error of '
							   'best model %f %%') %
							  (epoch, minibatch_index + 1, n_train_batches,
							   test_score * 100.))

						self.save(fpath=base_path + '/data/')

				if patience <= iter:
					done_looping = True
					break

		end_time = timeit.default_timer()
		logger.info('Optimization complete.')
		logger.info('Best validation score of %f %% obtained at iteration %i, '
			  'with test performance %f %%' %
			  (best_validation_loss * 100., best_iter + 1, test_score * 100.))
		print >> sys.stderr, ('The code for file ' +
							  os.path.split(__file__)[1] +
							  ' ran for %.2fm' % ((end_time - start_time) / 60.))

	def predict(self, data):
		"""
		the CNN expects inputs with Nsamples = self.batch_size.
		In order to run 'predict' on an arbitrary number of samples we
		pad as necessary.

		"""
		if isinstance(data, list):
			data = np.array(data)
		if data.ndim == 1:
			data = np.array([data])

		nsamples = data.shape[0]
		n_batches = nsamples//self.batch_size
		n_rem = nsamples%self.batch_size
		if n_batches > 0:
			preds = [list(self.predict_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
										   for i in range(n_batches)]
		else:
			preds = []
		if n_rem > 0:
			z = np.zeros((self.batch_size, self.im_width * self.im_height))
			z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
			preds.append(self.predict_wrap(z)[0:n_rem])
		
		return np.hstack(preds).flatten()
	
	def predict_proba(self, data):
		"""
		the CNN expects inputs with Nsamples = self.batch_size.
		In order to run 'predict_proba' on an arbitrary number of samples we
		pad as necessary.

		"""
		if isinstance(data, list):
			data = np.array(data)
		if data.ndim == 1:
			data = np.array([data])

		nsamples = data.shape[0]
		n_batches = nsamples//self.batch_size
		n_rem = nsamples%self.batch_size
		if n_batches > 0:
			preds = [list(self.predict_proba_wrap(data[i*self.batch_size:(i+1)*self.batch_size]))\
										   for i in range(n_batches)]
		else:
			preds = []
		if n_rem > 0:
			z = np.zeros((self.batch_size, self.n_in * self.n_in))
			z[0:n_rem] = data[n_batches*self.batch_size:n_batches*self.batch_size+n_rem]
			preds.append(self.predict_proba_wrap(z)[0:n_rem])
		
		return np.vstack(preds)
		
	def shared_dataset(self, data_xy):
		""" Load the dataset into shared variables """

		data_x, data_y = data_xy
		shared_x = theano.shared(np.asarray(data_x,
											dtype=theano.config.floatX))

		shared_y = theano.shared(np.asarray(data_y,
											dtype=theano.config.floatX))

		if self.output_type in ('binary', 'softmax'):
			return shared_x, T.cast(shared_y, 'int32')
		else:
			return shared_x, shared_y

	def __getstate__(self):
		""" Return state sequence."""
		
		#check if we're using ubc_AI.classifier wrapper, 
		#adding it's params to the state
		if hasattr(self, 'orig_class'):
			superparams = self.get_params()
			#now switch to orig. class (MetaCNN)
			oc = self.orig_class
			cc = self.__class__
			self.__class__ = oc
			params = self.get_params()
			for k, v in superparams.iteritems():
				params[k] = v
			self.__class__ = cc
		else:
			params = self.get_params()  #sklearn.BaseEstimator
		if hasattr(self, 'cnn'):
			weights = [p.get_value() for p in self.cnn.params]
		else:
			weights = []
		state = (params, weights)
		return state

	def _set_weights(self, weights):
		""" Set fittable parameters from weights sequence.

		Parameters must be in the order defined by self.params:
			W, W_in, W_out, h0, bh, by
		"""
		i = iter(weights)
		if hasattr(self, 'cnn'):
			for param in self.cnn.params:
				param.set_value(i.next())

	def __setstate__(self, state):
		""" Set parameters from state sequence.

		Parameters must be in the order defined by self.params:
			W, W_in, W_out, h0, bh, by
		"""
		params, weights = state
		#we may have several classes or superclasses
		for k in ['n_comp', 'use_pca', 'feature']:
			if k in params:
				self.set_params(**{k:params[k]})
				params.pop(k)

		#now switch to MetaCNN if necessary
		if hasattr(self,'orig_class'):
			cc = self.__class__
			oc = self.orig_class
			self.__class__ = oc
			self.set_params(**params)
			self.ready()
			if len(weights) > 0:
				self._set_weights(weights)
			self.__class__ = cc
		else:
			self.set_params(**params)
			self.ready()
			self._set_weights(weights)
	
	def save(self, fpath='.', fname=None):
		""" Save a pickled representation of Model state. """
		import datetime
		fpathstart, fpathext = os.path.splitext(fpath)
		if fpathext == '.pkl':
			# User supplied an absolute path to a pickle file
			fpath, fname = os.path.split(fpath)

		elif fname is None:
			# Generate filename based on date
			date_obj = datetime.datetime.now()
			date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
			class_name = self.__class__.__name__
			#fname = '%s.%s.pkl' % (class_name, date_str)
			fname = 'best_model.pkl'

		fabspath = os.path.join(fpath, fname)

		logger.info("Saving to %s ..." % fabspath)
		file = open(fabspath, 'wb')
		state = self.__getstate__()
		pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
		file.close()

	def load(self, path):
		""" Load model parameters from path. """
		logger.info("Loading from %s ..." % path)
		file = open(path, 'rb')
		state = pickle.load(file)
		self.__setstate__(state)
		file.close()

class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:`W`
	and bias vector :math:`b`. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""

	def __init__(self, input, n_in, n_out):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		self.W = theano.shared(value=np.zeros((n_in, n_out),
												 dtype=theano.config.floatX),
								name='W', borrow=True)
		# initialize the baises b as a vector of n_out 0s
		self.b = theano.shared(value=np.zeros((n_out,),
												 dtype=theano.config.floatX),
							   name='b', borrow=True)

		# compute vector of class-membership probabilities in symbolic form
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		# compute prediction as class whose probability is maximal in
		# symbolic form
		self.y_pred = T.argmax(self.p_y_given_x, axis=1)

		# parameters of the model
		self.params = [self.W, self.b]

		# L1 norm ; one regularization option is to enforce L1 norm to
		# be small
		self.L1 = 0
		self.L1 += abs(self.W.sum())

		# square of L2 norm ; one regularization option is to enforce
		# square of L2 norm to be small
		self.L2_sqr = 0
		self.L2_sqr += (self.W ** 2).sum()

	def negative_log_likelihood(self, y):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
				\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

	def errors(self, y):
		"""Return a float representing the number of errors in the minibatch
		over the total number of examples of the minibatch ; zero one
		loss over the size of the minibatch

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label
		"""

		# check if y has same dimension of y_pred
		if y.ndim != self.y_pred.ndim:
			raise TypeError('y should have the same shape as self.y_pred',
				('y', target.type, 'y_pred', self.y_pred.type))
		# check if y is of the correct datatype
		if y.dtype.startswith('int'):
			# the T.neq operator returns a vector of 0s and 1s, where 1
			# represents a mistake in prediction
			return T.mean(T.neq(self.y_pred, y))
		else:
			raise NotImplementedError()

class HiddenLayer(object):
	def __init__(self, rng, input, n_in, n_out, W=None, b=None,
				 activation=T.tanh):
		"""
		Typical hidden layer of a MLP: units are fully-connected and have
		sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
		and the bias vector b is of shape (n_out,).

		NOTE : The nonlinearity used here is tanh

		Hidden unit activation is given by: tanh(dot(input,W) + b)

		:type rng: np.random.RandomState
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
			W_values = np.asarray(rng.uniform(
					low=-np.sqrt(6. / (n_in + n_out)),
					high=np.sqrt(6. / (n_in + n_out)),
					size=(n_in, n_out)), dtype=theano.config.floatX)
			if activation == theano.tensor.nnet.sigmoid:
				W_values *= 4

			W = theano.shared(value=W_values, name='W', borrow=True)

		if b is None:
			b_values = np.zeros((n_out,), dtype=theano.config.floatX)
			b = theano.shared(value=b_values, name='b', borrow=True)

		self.W = W
		self.b = b

		lin_output = T.dot(input, self.W) + self.b
		self.output = (lin_output if activation is None
					   else activation(lin_output))
		# parameters of the model
		self.params = [self.W, self.b]

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: np.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height,filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows,#cols)

		"""

		assert image_shape[1] == filter_shape[1]
		self.input = input

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = np.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
				   np.prod(poolsize))
		# initialize weights with random weights
		W_bound = np.sqrt(6. / (fan_in + fan_out))
		self.W = theano.shared(
			np.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			),
			borrow=True
		)

		# the bias is a 1D tensor -- one bias per output feature map
		b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
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
		# reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# store parameters of this layer
		self.params = [self.W, self.b]

		self.input = input

def cosine_distance(a, b):
    import numpy as np
    from numpy import linalg as LA 
    dot_product =  np.dot(a,b.T)
    cosine_distance = dot_product / (LA.norm(a) * LA.norm(b))
    return cosine_distance

if __name__ == '__main__':
	
	base_path = '/Applications/MAMP/htdocs/DeepLearningTutorials' 
	#base_path = '/home/ubuntu/DeepLearningTutorials' 

	from fetex_image import FetexImage
	from PIL import Image
	import random

	datasets = load_data('mnist.pkl.gz')

	train_set_x, train_set_y = datasets[0]
	valid_set_x, valid_set_y = datasets[1]
	test_set_x, test_set_y = datasets[2]

	cnn = MetaCNN(learning_rate=0.05,nkerns=[48,128,256], filters=[13,5,4], batch_size=64,poolsize=[(2,2),(2,2),(2,2)], n_hidden=[200,50,2] , n_out=2, im_width=128,im_height=128)
	# cnn.fit(train_set_x,train_set_y,valid_set_x,valid_set_y,test_set_x,test_set_y, n_epochs=5)
	# cnn.save(fpath=base_path + '/data/')


	#folder = base_path + '/data/cnn-furniture/'

	# Predictions after training
	cnn.load(base_path + '/data/best_model.pkl')
	#cnn.load('/home/ubuntu/DeepLearningTutorials/data/MetaCNN.2015-10-19-13:59:18.pkl')
	#sample = np.asarray(X_train, dtype=theano.config.floatX)
	#print sample[0].reshape((64,64)).shape
	#Image.fromarray(sample[2].reshape((64,64)),mode="L").show()

	pkl_file = open( '../data/train_set.pkl', 'rb')
	train_set = pickle.load(pkl_file)

	X_train, Y_train = train_set

	pkl_file = open( '../data/lb.pkl', 'rb')
	lb = pickle.load(pkl_file)

	# arr = np.array(np.round((X_train[0] * 256).reshape((128,128))),dtype=np.uint8)
	# Image.fromarray(arr,mode="L").show()

	# arr = np.array(np.round((X_train[1] * 256).reshape((128,128))),dtype=np.uint8)
	# Image.fromarray(arr,mode="L").show()

	# arr = np.array(np.round((X_train[2] * 256).reshape((128,128))),dtype=np.uint8)
	# Image.fromarray(arr,mode="L").show()
	
	#print Y_train[0:3]
	# arr = np.array(np.round((X_train[1300] * 256).reshape((64,64))),dtype=np.uint8)
	# Image.fromarray(arr,mode="L").show()
	#print sample[0]
	# #print sample.shape
	#sample = X_train[0:25]
	#print lb.classes_ 
	#sample = X_train[0]
	#print Y_train[4000:4100]
	#print cnn.predict(X_train[0:3])

	# sample = X_train[4400]
	# print Y_train[4400]
	# print cnn.predict(sample)
	# pkl_file = open( '../data/X_original.pkl', 'rb')
	# X_original = cPickle.load(pkl_file)

	# a = X_original[0:25]
	# a = np.asarray(a, dtype=theano.config.floatX)
	# #fe.reconstructImage(a[2]).show()

	def flaten_aux(V):
	    return V.flatten(order='F')

	#print X_train[0].shape

	# cnn_output_vectors = np.array([])
	# for i in xrange(1,8):
	# 	#a = map(flaten_aux, X_train[128 * (i - 1): 128 * i ])
	# 	a = X_train[64 * (i - 1): 64 * i ]
	# 	# #print cnn.predict(a)
	# 	a = cnn.predict_vector(a)
	# 	#print a
	# 	print len(cnn_output_vectors)
	# 	#cnn_output_vectors.append(a)
	# 	if len(cnn_output_vectors) == 0:
	# 		cnn_output_vectors = a
	# 	else:
	# 		cnn_output_vectors = np.concatenate((cnn_output_vectors, a), axis=0)
	# 		#cnn_output_vectors = cnn_output_vectors + a

	# print len(cnn_output_vectors)

	# file = open('../data/cnn_output_vectors.pkl', 'wb')
	# pickle.dump(cnn_output_vectors, file, protocol=pickle.HIGHEST_PROTOCOL)
	# file.close()

	file = open('../data/cnn_output_vectors.pkl', 'rb')
	cnn_output_vectors = pickle.load(file)
	file.close()
	print len(cnn_output_vectors)
	#print len(cnn_output_vectors)
	#print len(X_train)
	#print cnn.predict(sample)
	#print cnn.predict_wrap(a)
	#rn_im_index  = random.randint(0, len(X_train))
	#base_image_index  = 1
	base_image_index  = random.randint(0, 448)
	max_similarity = 0
	max_similarity_pos = -1
	#max_similarity_pos = []
	#for i in xrange(1,len(train_set_x)):
	a = cnn_output_vectors[base_image_index]
	#a = X_train[base_image_index]
	#print a.shape
	for i in xrange(0,64 * 7):
		
		if i != base_image_index:
			b = cnn_output_vectors[i]
			#b = X_train[i]
			d = cosine_distance(a, b)
			print d
			#if d > max_similarity:
			if d > max_similarity:
				max_similarity = d
				max_similarity_pos = i
				#max_similarity_pos.append(i)

	print 'max_similarity: {}'.format(max_similarity)
	fe = FetexImage(mode='L')
	fe.reconstructImage(X_train[base_image_index]).show()
	fe.reconstructImage(X_train[max_similarity_pos]).show()
	# fe.reconstructImage(X_train[max_similarity_pos[0]]).show()
	# fe.reconstructImage(X_train[max_similarity_pos[1]]).show()
	# fe.reconstructImage(X_train[max_similarity_pos[2]]).show()
	# fe.reconstructImage(X_train[max_similarity_pos[3]]).show()
	# print a.shape
	# print b.shape
	# print cosine_distance(a, b)