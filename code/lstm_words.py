'''
Build a tweet sentiment analyzer
'''
from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import imdb
import random
#config.mode = 'DebugMode'

config.floatX = 'float64'
datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['n_words'],
                              options['dim_proj'])
    params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = get_layer(options['encoder'])[0](options,
                                              params,
                                              prefix=options['encoder'])
    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    W = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_layer(tparams, state_below, options, prefix='lstm', x=None):
    nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert x is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):

        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # x_printed = theano.printing.Print('this is a very important value')(preact)

        # f = theano.function([preact], preact)
        # f_with_print = theano.function([preact], x_printed)

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    #print state_below
    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    
    #print config.floatX

    rval, updates = theano.scan(_step,
                                sequences=[x, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0],nsteps


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer)}


def sgd(lr, tparams, grads, x, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, y, cost):
    """
    An adaptive learning rate optimizer

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    # f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
    #                                 name='adadelta_f_grad_shared', allow_input_downcast=True)
    #print type(x)
    #cost = theano.function([x],x,on_unused_input='warn')
    # f_grad_shared = theano.function([x], x, name='adadelta_f_grad_shared',on_unused_input='warn')
    f_grad_shared = theano.function([x, y], cost, updates=zgup + rg2up,
                                   name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, y, cost):
    """
    A variant of  SGD that scales the step size by running average of the
    recent step norms.

    Parameters
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [Hint2014]_.

    .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
       lecture 6a,
       http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    """

    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['dim_proj']])
    proj,nsteps = get_layer(options['encoder'])[1](tparams, emb, options,
                                            prefix=options['encoder'],
                                            x=x)

    if options['encoder'] == 'lstm':
        proj = (proj * x[:, :, None]).sum(axis=0)
        proj = proj / x.sum(axis=0)[:, None]
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x], pred, name='f_pred_prob')
    f_pred = theano.function([x], pred.argmax(axis=1), name='f_pred')
    
    get_nsteps = theano.function([x], nsteps, name='get_nsteps',on_unused_input='warn')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, y, f_pred_prob, f_pred, cost,get_nsteps

def pred_probs(f_pred_prob, prepare_data, data, iterator, dic, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    voc_size = len(dic['words2idx'].keys()) + 1
    
    print "n_samples {} voc_size {}".format(n_samples,voc_size)
    #print numpy.max(data[1])
    #probs = numpy.zeros((n_samples, 2)).astype(config.floatX)
    probs = numpy.zeros((n_samples, voc_size)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)

        pred_probs = f_pred_prob(x)
        probs[valid_index, :] = pred_probs
        n_done += len(valid_index)
        if verbose:
            print '%d/%d samples classified' % (n_done, n_samples)

    return probs

def samples(f_pred,x ,dic , verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(x)
    voc_size = len(dic['words2idx'].keys()) + 1
    
    #print "n_samples {} voc_size {}".format(n_samples,voc_size)
    #print x
    
    #x_ixes = [[x[0][0]],[x[1][0]]]
    #x_ixes = x
    #x_ixes = [list(x[0])]
    #print x
    x = [x[0][0:10]]
    
    x_ixes = list(x[0])
    
    x = [[word] for word in x[0]]
    #print x
    #x_ixes = [list(x[0])]
    #x_ixes = list(x[0])
    #x_aux = list(x)
    for i in xrange(0,10):
        #pred_probs = f_pred_prob(x)
        predicted = f_pred(x)[0]
        print predicted
        #print predicted
        #predicted = numpy.random.choice(range(voc_size), p=pred_probs.ravel())

        #x_ixes.append(predicted)
        #print x_ixes
        #x_ixes[0].append(predicted)
        x_ixes.append(predicted)
        #print x_ixes

        x = numpy.zeros((len(x_ixes),1),dtype='int64')
        #x = numpy.zeros(len(x_ixes),dtype='int64')
        #x[len(x_ixes) - 1] = predicted
        #print x
        #x = []
        
        #for j in xrange(0,len(x_ixes[0])):
        for j in xrange(0,len(x_ixes)):
            #print x_ixes[j]
            x[j] = [x_ixes[j]]
            #x[j] = x_ixes[0][j]
            #x[j] = x_ixes[j]
            #x.append(x_ixes[j])
        #x = [x]
        #print x

        

    sample = " ".join([dic['idx2word'][w_ix] for w_ix in x_ixes])
    #sample = " ".join([dic['idx2word'][w_ix] for w_ix in x_ixes[0]])
    print sample

    #return probs

def pred_error(f_pred, prepare_data, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=None)
        preds = f_pred(x)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])

    return valid_err

def remove_return_lines_and_quotes(text):
        text = text.replace('\n', ' ')
        text = text.replace('\t', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('"', '')
        text = text.replace('-', '')
        return text

def load_text():
    import nltk
    import numpy as np
    from nltk.corpus import gutenberg
    import csv
    from bs4 import BeautifulSoup

    # with open('product_description_clean.csv', 'rU') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',', quotechar='"', dialect=csv.excel_tab)
    #     next(reader, None)  # skip the headers
    #     documents = []
    #     err_docs = 0
    #     for row in reader:
    #         try:
    #             #d = BeautifulSoup(row[0]).getText()
    #             d = BeautifulSoup(row[0], "html.parser").getText()
    #             documents.append(d)
    #         except Exception, e:
    #             #print e
    #             err_docs +=1
    # print "there are {} errors".format(err_docs)

    dic = {'labels2idx' : {} , 'words2idx' : {}}
    classes = {}

    #text = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.'
    print '...loading corpus'
    text = gutenberg.raw()
    #text = " ".join(documents)
    text = text[0:1000000]
    # text = text[0:100] + '.'
    # print text

    #text = text[0:1000000]
    text = remove_return_lines_and_quotes(text)
    #print "text lengh:{}".format(len(documents))
    #print text

    #text = self.remove_return_lines_and_quotes(text)
    print "...tokenizing"
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]

    print "...creating indexes"
    for sentence in sentences:
        for word in sentence:
            dic['words2idx'][word.lower()] = 1

    for i in range(0,len(dic['words2idx'].keys())):
        key = dic['words2idx'].keys()[i]
        dic['words2idx'][key] = i + 1

    sentences_words,sentences_ne,sentences_labels = [],[],[]
    words , labels = [],[]
    for sentence in sentences:
        for i in range(0,len(sentence)):
            words.append([dic['words2idx'][sentence[i].lower()]])

            if (i + 1) == len(sentence):
                labels.append(dic['words2idx']['.'])
            else:
                labels.append(dic['words2idx'][sentence[i+1].lower()])

    print "...creating training data"
    # X = np.array(sentences_words)
    # Y = np.array(sentences_labels)
    X = np.array(words, dtype='int64')
    Y = np.array(labels, dtype='int64')

    train_length = int(round(len(X) * 0.90))
    valid_length = int(round(len(X) * 0.05))
    test_length = int(round(len(X) * 0.05))

    X_train = X[0:train_length]
    X_valid = X[train_length: (train_length + valid_length)]
    X_test = X[-test_length:]

    Y_train = Y[0:train_length]
    Y_valid = Y[train_length: (train_length + valid_length)]
    Y_test = Y[-test_length:]

    train = [X_train,Y_train]
    valid = [X_valid,Y_valid]
    test = [X_test,Y_test]

    return train,valid,test,dic

def load_sequences():
    import nltk
    import numpy as np
    from nltk.corpus import gutenberg
    import csv
    from bs4 import BeautifulSoup
    from sys import getsizeof

    dic = {'labels2idx' : {} , 'words2idx' : {}}
    classes = {}

    #text = 'Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.'
    print '...loading corpus'
    f = open('../data/text_input.txt', 'r')
    text = f.read()
    text = text.decode('utf-8','replace')
    #print text
    f.close()
    text = gutenberg.raw()
    
    #text = " ".join(documents)
    #text = text[0:4000000]
    #text = text[0:1000000]
    text = text[0:100000]
    print getsizeof(text)
    #text = text[0:100] + '.'
    # print text

    #text = text[0:1000000]
    text = remove_return_lines_and_quotes(text)
    #print "text lengh:{}".format(len(documents))
    #print text

    #text = self.remove_return_lines_and_quotes(text)
    print "...tokenizing"
    sentences = nltk.sent_tokenize(text)
    sentences = [nltk.word_tokenize(sent) for sent in sentences]

    print "...creating indexes"
    text_words = []
    for sentence in sentences:
        # print sentence
        # print "sentence_length:{}".format(len(sentence))
        for word in sentence:
            dic['words2idx'][word.lower()] = 1
            text_words.append(word)

    for i in range(0,len(dic['words2idx'].keys())):
        key = dic['words2idx'].keys()[i]
        dic['words2idx'][key] = i + 1

    sequence_length = 1
    num_sequences = len(text_words) / sequence_length

    words , labels = [],[]
    #for sentence in sentences:
    #for sentence in sentences:
    print num_sequences
    #for i in range(0,int(num_sequences)):
    for i in range(0,len(text_words)):
        #sequence = text_words[ i * sequence_length : (i+1) * sequence_length]
        sequence = text_words[ i: i + sequence_length]
        #print sequence
        words_aux = []
        for item in sequence:
            #print item
            #print dic['words2idx'][item.lower()]
            words_aux.append(dic['words2idx'][item.lower()])
        #print "--"
        #print text_words[ ((i+1) * sequence_length)]
        words.append(words_aux)
        #labels.append(dic['words2idx'][text_words[ i+1 ].lower()])
        if (i + 1) == len(text_words):
            labels.append(dic['words2idx']['.'])
        else:
            labels.append(dic['words2idx'][text_words[i+1].lower()])
    #print 
    # print words
    # print labels
    print "...creating training data"
    # X = np.array(sentences_words)
    # Y = np.array(sentences_labels)
    #X = np.array(words, dtype='int64')
    #Y = np.array(labels, dtype='int64')
    X = words
    Y = labels

    train_length = int(round(len(X) * 0.90))
    valid_length = int(round(len(X) * 0.05))
    test_length = int(round(len(X) * 0.05))


    X_train = X[0:train_length]
    X_valid = X[train_length: (train_length + valid_length)]
    X_test = X[-test_length:]

    Y_train = Y[0:train_length]
    Y_valid = Y[train_length: (train_length + valid_length)]
    Y_test = Y[-test_length:]

    train = [X_train,Y_train]
    valid = [X_valid,Y_valid]
    test = [X_test,Y_test]

    return train,valid,test,dic

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win/2 * [-1] + l + win/2 * [-1]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def contextwinright(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    #assert (win % 2) == 0
    assert win >=1
    l = list(l)

    #lpadded = win/2 * [-1] + l + win/2 * [-1]
    lpadded = win/2 * [-1] + win/2 * [-1] + l 
    #lpadded = win/2 * [-1] + l 
    #print lpadded
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def context(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    #assert (win % 2) == 1
    #assert (win % 2) == 0
    assert win >=1
    
    l = list(l)
    out = []
    for i in range(0,len(l)):

        if i < win:
            #s = l[0 : i + 1 ]
            if i == 0:
                s = l[0 : i + 1 ]
                #s = [1] + l[0 : i + 1 ]
            #else:
             #   s = l[0 : i + 1 ]
        else:
            s = l[ (i - 1) : (i - 1) + win]        

        #s = l[ (i - 1) : (i - 1) + win]        
        #s = numpy.array(s,dtype='int64')
        #s = [ [i] for i in s]
        s = numpy.array(s)
        #out.append([s])
        out.append(s)

    assert len(out) == len(l)

    out = numpy.array(out)

    return out

def train_lstm(
    dim_proj=256 * 2,  # word embeding dimension and LSTM number of hidden units.
    patience=30,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    n_words=10000,  # Vocabulary size
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=100,  # Sequence longer then this get ignored
    batch_size=32,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='imdb',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model=None,  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):
    
    #print context(range(0,16), 4)
    #return 1
    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset)

    print 'Loading data'
    # train, valid, test = load_data(n_words=n_words, valid_portion=0.05,
    #                                maxlen=maxlen)

    # print train[0][0]
    # print train[1][0]
    #print numpy.max(train[1]) + 1

    #train,valid,test,dic = load_text()
    train,valid,test,dic = load_sequences()

    #print train[1]
    #print len(dic['words2idx'].keys())
    #print train[0][0]
    #print train[1][0]
    #print train[0][0]
    #train[0][0] = [16]
    #train[0][0] = []
    #train[1][0] = 1
    # ydim = 0
    # for y in train[1]:
    #     if numpy.max(y) > ydim:
    #         ydim = numpy.max(y)
    # ydim += 1
    # print ydim

    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(test[0]))
        numpy.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])

    ydim = numpy.max(train[1]) + 1
    #print ydim

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)

    if reload_model:
        load_params('lstm_model.npz', params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x,
     y, f_pred_prob, f_pred, cost,get_nsteps) = build_model(tparams, model_options)

    # print train[0][0:10]
    # print train[1][0:10]
    dic['idx2word'] = dict((k, v) for v, k in dic['words2idx'].iteritems())
    print "vocabulary size: {}".format(len(dic['idx2word'].keys()))

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, y, cost)

    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])

    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            # Get new shuffled index for the training set.
            #kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=False)
            #print len(train[0])
            #print kf
            for _, train_index in kf:
                #print "--------------------------------"
                #print train_index
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch                
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]
                #x = [train[0][t][0] for t in train_index]
                #print x
                # print len(x)
                #x = contextwin(x, 7)
                #x = contextwinright(x, 3
                # TODO: each input in x must be the same length. I thought LSTM were capable of managing multiple lenght inputs
                # x = context(x, 1)
                #x = numpy.array(x)
                #print x.shape
                # y = numpy.array(y)
                # print len(x)
                # print len(x[0])
                # print len(x[1])

                #print x
                #print len(x)
                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                #x = [train[0][t] for t in train_index]
                #x, mask, y = prepare_data(x, y)
                
                #print x.shape
                # print type(x)
                # print x[0].shape
                # print type(x[0])
                # print x[0]
                # print x[3].shape
                # print x[4].shape
                # print x[3:4]
                # print y

                #print " ".join([dic['idx2word'][w_ix] for w_ix in x[0]])
                #x = [ contextwin(sample, 7) for sample in x]
                #print x[0]
                #n_samples += x.shape[1]

                
                # x = range(1,8)
                # x = []
                #x = map(numpy.array, x)
                #print x
                # x = x[
                #     numpy.array([1]),
                #     numpy.array([2]),
                #     numpy.array([3]),
                #     numpy.array([4]),
                #     numpy.array([5]),
                #     numpy.array([6]),
                #     numpy.array([7])
                # ]


                # x = [[1],[2],[6],[10],[3],[4],[5],[7]]
                x = numpy.array(x).T
                # y = [2,6,10,3,4,5,7,8]

                # print x
                # print y
                #print x
                #print get_nsteps([[1]])

                #y = numpy.array(range(6,11))
                # y = numpy.array([0,1])
                #print x.shape

                n_samples += x.shape[0]

                cost = f_grad_shared(x,y)
                #print nsteps.get_value()
                #print get_nsteps([[1]])
                #print get_nsteps([1]).get_value()

                #print cost.get_value().shape
                # print "costicontes"
                # print cost
                f_update(lrate)
                
                # if numpy.mod(uidx, validFreq) == 0:
                #     #pred_x = numpy.array(numpy.array([0,1]))
                #     #pred_x = numpy.array([numpy.array([1,2])]) 
                #     pred_x = numpy.array([numpy.array([3])]) 
                #     print pred_x
                #     predicted_x = f_pred(pred_x)

                #     print predicted_x

                #     for z in range(0,5):
                #         pred_x = numpy.array([numpy.array(predicted_x)]) 
                #         #print pred_x
                #         predicted_x = f_pred(pred_x)
                #         print predicted_x

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.
                    #continue

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost
                    #samples(f_pred,[[x[0][0]]],dic)

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    
                    start_time_2 = time.time()

                    use_noise.set_value(0.)
                    print "train_error"
                    print len(kf)
                    train_err = pred_error(f_pred, prepare_data, train, kf)
                    print "train_err"
                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    print "valid_error"
                    test_err = pred_error(f_pred, prepare_data, test, kf_test)
                    print "test_err"
                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    samples(f_pred,x,dic)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break
                    
                    end_time_2 = time.time()
                    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time_2 - start_time_2))

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    train_lstm(
        max_epochs=1000,
        test_size=1000,
        reload_model=None
    )
