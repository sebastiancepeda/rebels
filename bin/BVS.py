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
import json
import scipy.io as sio
###############################
def build_custom_mlp(n_features, input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
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

# ############################## Main program ################################
def main():
    print('* Start! ')
    print('* Loading config.json')
    with open('config.json') as json_data:
        config = json.load(json_data)
        depth = int(config["layers"])
        width = int(config["neurons_by_layer"])
        drop_in = float(config["dropout_input"])
        drop_hid = float(config["dropout_hidden"])
        num_epochs = int(config["num_epochs"])    
        winSide = int(config["window_side"])    
        ImageShape = config["image_shape"]
        ImageShape = (int(ImageShape[0]),int(ImageShape[1]),int(ImageShape[2]))
        alpha = float(config["alpha"])
    # Other global variables
    PatternShape   	= (ImageShape[0],ImageShape[1])
    winSize        	    = (winSide,winSide)
    n_features     	    = ImageShape[2]*(winSide**2)
    # Loading dataset
    X_train, y_train, mask = utils.load_data(ImageShape, PatternShape, winSize)
    mask = 0
    print("* Building model and compiling functions...")
    input_var = T.dmatrix('inputs')
    target_var = T.ivector('targets')
    network = build_custom_mlp(n_features, input_var, depth, width, drop_in, drop_hid)
    prediction = lasagne.layers.get_output(network)
    t2 = theano.tensor.extra_ops.to_one_hot(target_var, 2, dtype='int32')
    error = lasagne.objectives.categorical_crossentropy(prediction, t2)
    loss = error.mean()/n_features
    params = lasagne.layers.get_all_params(network, trainable=True)
    train_fn = theano.function([input_var, target_var], [loss])
    output_model = theano.function([input_var], prediction)
    # grads compilation
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
        return (x_out, y_out,  0)
        
    def getAUC(w_t,  X_train, y_train):
        params_updater(w_t)
        m_data = X_train.shape[0]
        output_train = numpy.zeros((m_data, 2))
        points  = numpy.floor(numpy.linspace(0,m_data,m_data/1000)).astype(int)
        for it in numpy.arange(points.size-1):
            output_train[points[it]:points[it+1], :] = output_model(X_train[points[it]:points[it+1], :])
        out = output_train[:,1]
        auc = roc_auc_score(y_train, out)
        return auc
    
    def optimizer(func, x0, fprime, training_data, callback):
        print('* Optimizer method. ')
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
            if((i > 10) and (i % 50 == 0)):
                numpy.save('../data/w_t.npy',w_t)
                sio.savemat('../data/BVS_data.mat', {'depth':depth,'width':width,'drop_in':drop_in,'drop_hid':drop_hid,'w_t':w_t})
                auc_it[it2] = getAUC(w_t,  X_train, y_train)
                print('AUC: {}'.format(auc_it[it2]))
                auc_x[it2] = i
                it2 += 1
            if((i > 10) and (i % 50 == 0)):
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(numpy.arange(i), e_it[0:i], 'r-')
                fig.savefig('debug/error.png')
                plt.close(fig)
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(numpy.arange(i), de_it[0:i], 'g-')
                fig.savefig('debug/dw_t.png')
                plt.close(fig)
                fig, ax = plt.subplots(nrows=1, ncols=1)
                ax.plot(auc_x[0:it2], auc_it[0:it2], 'b-')
                fig.savefig('debug/auc.png')
                plt.close(fig)
                print('Show test imge... ')
                y_preds  = output_model(X_train)
                output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
                aux_image = output_image
                img = numpy.floor(output_image*255)
                cv2.imwrite('debug/image-last.png',img)
    
    optimizer(func, x0=params_giver(), fprime=fprime, training_data=(X_train,y_train.astype(numpy.int32)),  callback=None)
    
    print('* Show test images... ')
    for i in numpy.arange(21, 41):
        X_train, y_train = utils.load_data(ImageShape, PatternShape, winSize, i)
        y_preds  = output_model(X_train)
        output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
        aux_image = output_image
        img = numpy.floor(output_image*255)
        print('Test image: {}'.format(i))
        cv2.imwrite('debug/y_preds-'+str(i)+'.png',img)

if __name__ == '__main__':
    main()
