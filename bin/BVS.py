from __future__ import print_function
from __future__ import division
import sys
import random
import os
import time
import numpy as np
import numpy
import theano
import theano.tensor as T
import lasagne
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob, os
import csv
import sqlite3
import time
import datetime
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score
import utils
###############################
print('Start! ')
it_counter = 0
ImageShape     = (584,565,3)
PatternShape   = (584,565)
winSize            = (15, 15)
n_features       = 3*winSize[0]*winSize[1]
alpha               = 1
###############################
def load_data(n_image = 21):
    print('Reading data. ')
    features 	   = Image.open('../DRIVE/training/images/'+str(n_image)+'_training.tif','r')    
    features     = numpy.fromstring(features.tobytes(), dtype='uint8', count=-1, sep='')
    features     = numpy.reshape(features,ImageShape,'A')
    image2     = Image.open('../DRIVE/training/1st_manual/'+str(n_image)+'_manual1.gif','r')
    image2     = numpy.fromstring(image2.tobytes(), dtype='uint8', count=-1, sep='')
    image2     = numpy.reshape(image2,PatternShape,'A')
    train_set  = utils.sliding_window(features, stepSize=1, w=winSize, dim=ImageShape[2],output=0)
    train_set = train_set.reshape(train_set.shape[0], numpy.floor(train_set.size/train_set.shape[0]).astype(int))
    train_set_t  = utils.sliding_window(image2, stepSize=1, w=winSize, dim=1,output=1)
    train_set_t = train_set_t[:,0]
    print('Scaling data. ')
    train_set = preprocessing.scale(train_set)
    return train_set, train_set_t

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    network = lasagne.layers.InputLayer(shape=(None, n_features), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
    #nonlin = lasagne.nonlinearities.very_leaky_rectify
    nonlin = lasagne.nonlinearities.leaky_rectify
    for _ in range(depth):
        network = lasagne.layers.DenseLayer(network, width, nonlinearity=nonlin)
        if drop_hidden:
            network = lasagne.layers.dropout(network, p=drop_hidden)
    # Output layer:
    last_nonlin = lasagne.nonlinearities.softmax
    network = lasagne.layers.DenseLayer(network, 2, nonlinearity=last_nonlin)
    return network

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ############################## Main program ################################
def main(model='mlp', num_epochs=500):
    X_train, y_train = load_data()
    input_var = T.dmatrix('inputs')
    target_var = T.ivector('targets')
    p = T.dvector('p')
    print("Building model and compiling functions...")
    if model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width), float(drop_in), float(drop_hid))
    else:
        print("Unrecognized model type %r." % model)
        return
    prediction = lasagne.layers.get_output(network)
    t2 = theano.tensor.extra_ops.to_one_hot(target_var, 2, dtype='int32')
    error = lasagne.objectives.categorical_crossentropy(prediction, t2)
    loss = error.mean()/n_features# + lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)*0.0000000001
    params = lasagne.layers.get_all_params(network, trainable=True)
    train_fn = theano.function([input_var, target_var], [loss])
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = ((test_prediction[:, 1]-target_var)*(test_prediction[:, 1]-target_var)).mean()
    test_loss = test_loss.mean()
    output_model = theano.function([input_var], prediction)
    # grads
    grads = []
    for w in params:
        grad = T.grad(loss,  w)
        grad_fn = theano.function([input_var, target_var], grad)
        grads = grads + [grad_fn]
    
    def params_giver():
        ws = []
        for w in params:
            params_fn = theano.function([],  w)
            w_out = params_fn()
            ws = numpy.append(ws,  w_out)
        return ws
    
    def grad_giver(x,  t,  p_data):
        gs = []
        for grad_fn in grads:
            gs = numpy.append(gs,  grad_fn(x,  t))
        return gs
    '''
    Method that receives the new set of weights 
    and inserts them in the net. 
    '''
    def params_updater(all_w):
        idx_init = 0
        for w in params:
            w_in = T.dmatrix()
            if(w_in.type != w.type):
                w_in = T.dvector()
            w_update = theano.function([w_in], updates=[(w, w_in)])
            w_value_pre = w.get_value()
            w_act = all_w[idx_init:idx_init+w_value_pre.size]
            w_value = w_act.reshape(w_value_pre.shape)
            idx_init += w_value_pre.size
            w_update(w_value)
        return
    
    '''
    Method that receives the weights (params), training data (args), 
    and returns the loss on the data. 
    It also receives the new set of weights 
    and inserts them in the net. 
    '''
    def func(params, *args):
        x = args[0]
        t = args[1]
        if(len(args) >= 3):
            p_data = args[2]
        params_updater(params)
        return train_fn(x,  t.astype(numpy.int32))[0]
    
    '''
    Method that receives the weights (params), training data (args)
    and returns the derivative of the loss on the data with 
    respect to the weights. 
    '''
    def fprime(params, *args):
        x = args[0]
        t = args[1]
        if(len(args) >= 3):
            p_data = args[2]
        else:
            p_data = 1
        params_updater(params)
        return grad_giver(x, t.astype(numpy.int32), p_data)
    
    def sampleData(training_data,  p_total,  n_samples = 10000):
        x = training_data[0]
        y = training_data[1]
        inds = random.sample(range(training_data[0].shape[0]), n_samples)
        x_out = x[inds,:]
        y_out = y[inds]
        #print('x.shape: {}'.format(x.shape))
        return (x_out, y_out,  0)
        
    def getAUC(w_t,  X_train, y_train):
        print('w_t.shape: {}'.format(w_t.shape))
        params_updater(w_t)
        m_data = X_train.shape[0]
        output_train = numpy.zeros((m_data, 2))
        points  = numpy.floor(numpy.linspace(0,m_data,m_data/1000)).astype(int)
        for it in numpy.arange(points.size-1):
            output_train[points[it]:points[it+1], :] = output_model(X_train[points[it]:points[it+1], :])
            #print('it: {}/{}'.format(it, points.size))
        out = output_train[:,1]
        auc = roc_auc_score(y_train, out)
        print('AUC: {}'.format(auc))
        return auc
    
    def optimizer(func, x0, fprime, training_data, callback):
        print('Optimizer method. ')
        n_samples = 1000
        w_t = x0
        m_t = 0
        p = 1
        train_data = sampleData(training_data,  p,  n_samples = n_samples)
        args = (train_data[0], train_data[1])
        e_t = func(w_t , *args)
        e_it = numpy.zeros(num_epochs)
        de_it = numpy.zeros(num_epochs)
        auc_it = numpy.zeros(num_epochs)
        auc_x = numpy.zeros(num_epochs)
        m_r = 0.99
        l_start = 1
        l_end = 0.01
        it2 = 0
        for i in numpy.arange(num_epochs):
            dedw = fprime(w_t , *args)
            g_t = -dedw
            l_r = l_end+(l_start-l_end)*(0.99**(i/10))
            m_t = m_r*m_t + g_t*l_r
            dw_t  = m_r*m_t + g_t*l_r
            w_t = w_t + dw_t
            e_t = func(w_t , *args)
            e_it[i] = e_t
            if(i % 10 == 0):
                train_data = sampleData(training_data,  p,  n_samples = n_samples)
                args = (train_data[0], train_data[1])
                print("i: {}, e_t: {}, l_r: {}".format(i, e_t,  l_r))
            de_it[i] = numpy.abs(dw_t).mean()
            #print("numpy.abs(m_t).mean(): {}".format(numpy.abs(m_t).mean()))
            if((i > 10) and (i % 50 == 0)):
                numpy.save('../data/w_t.npy',w_t)
                auc_it[it2] = getAUC(w_t,  X_train, y_train)
                auc_x[it2] = i
                it2 += 1
            if((i > 10) and (i % 50 == 0)):
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(numpy.arange(i), e_it[0:i], 'r-')
                fig.savefig('error.png')
                plt.close(fig)
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(numpy.arange(i), de_it[0:i], 'g-')
                fig.savefig('dw_t.png')
                plt.close(fig)
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(auc_x[0:it2], auc_it[0:it2], 'b-')
                fig.savefig('auc.png')
                plt.close(fig)
                print('Show test imge... ')
                y_preds  = output_model(X_train)
                output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
                aux_image = output_image
                img = numpy.floor(output_image*255)
                cv2.imwrite('debug/image-last.png',img)
    
    optimizer(func, x0=params_giver(), fprime=fprime, training_data=(X_train,y_train.astype(numpy.int32)),  callback=None)
    
    print('Show test images... ')
    for i in numpy.arange(21, 40):
        X_train, y_train = load_data(i)
        y_preds  = output_model(X_train)
        output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
        aux_image = output_image
        img = numpy.floor(output_image*255)
        print('Test image: {}'.format(i))
        cv2.imwrite('debug/y_preds-'+str(i)+'.png',img)

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on RedHat using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
