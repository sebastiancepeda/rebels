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
	learning_rate = float(config["learning_rate"])
    # Other global variables
    PatternShape   	= (ImageShape[0],ImageShape[1])
    winSize        	    = (winSide,winSide)
    n_features     	    = ImageShape[2]*(winSide**2)
    
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
    
    def sampleData(valid_windows,n_samples,x_image,  t_image,winSize,ImageShape,x_mean, x_std):
        inds = random.sample(range(valid_windows), n_samples)
        #inds = range(valid_windows)
        x, t = utils.sample_sliding_window(x_image,  t_image,winSize,ImageShape[2],x_mean, x_std,  inds)
        x = x.reshape(x.shape[0], numpy.floor(x.size/x.shape[0]).astype(int))
        t = t.astype(numpy.int32)
        return (x, t)
    
    def getAUC(w_t,  y_preds,  y_train):
        params_updater(w_t)
        auc = roc_auc_score(y_train, y_preds[:, 1])
        return auc
    
    # Loading dataset
    n1 = ImageShape[0]
    n2 = ImageShape[1]
    diff = (winSize[0]-1)//2
    valid_windows = int(n1*n2-diff*2*(n1+n2)+4*diff*diff)
    x_image,  t_image,  mask_image = utils.get_images(ImageShape, PatternShape, winSize, 21)
    x_mean = utils.get_mean(x_image,  winSize,  ImageShape[2],  ImageShape)
    x_std = utils.get_std(x_image,  winSize,  ImageShape[2],  ImageShape,  x_mean)
    
    def optimizer(func, x0, fprime, training_data, callback):
        print('* Optimizer method. ')
        n_samples = 1000
        w_t = x0
        m_t = 0
        train_data = sampleData(valid_windows,n_samples,x_image,  t_image,winSize,ImageShape,x_mean, x_std)
        e_t = func(w_t , *train_data)
        e_it = numpy.zeros(num_epochs)
        de_it = numpy.zeros(num_epochs)
        auc_it = numpy.zeros(num_epochs)
        auc_x = numpy.zeros(num_epochs)
        m_r = 0.99
        it2 = 0
        for i in numpy.arange(num_epochs):
            dedw = fprime(w_t , *train_data)
            g_t = -dedw
            l_r = learning_rate
            m_t = m_r*m_t + g_t*l_r
            dw_t  = m_r*m_t + g_t*l_r
            w_t = w_t + dw_t
            e_t = func(w_t , *train_data)
            e_it[i] = e_t
            if(i % 10 == 0):
                train_data = sampleData(valid_windows,n_samples,x_image,  t_image,winSize,ImageShape,x_mean, x_std)
                print("i: {}, e_t: {}, time: {}".format(i, e_t, time.ctime()))
            de_it[i] = numpy.abs(dw_t).mean()
            if((i > 10) and (i % 100 == 0)):
                numpy.save('../data/w_t.npy',w_t)
                sio.savemat('../data/BVS_data.mat', {'depth':depth,'width':width,'drop_in':drop_in,'drop_hid':drop_hid,'w_t':w_t})
                y_preds = utils.get_predictions(x_image, ImageShape, PatternShape, winSize, output_model,  x_mean, x_std)
                t_data = utils.sliding_window(t_image, winSize, dim=1,output=1)
                auc_it[it2] = getAUC(w_t,  y_preds,  t_data)
                print('AUC: {}'.format(auc_it[it2]))
                auc_x[it2] = i
                it2 += 1
                # debug images
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
                output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
                img = numpy.floor(output_image*255)
                cv2.imwrite('debug/image-last.png',img)
    
    optimizer(func, x0=params_giver(), fprime=fprime, training_data=None,  callback=None)
    
    print('* End of optimization. ')

if __name__ == '__main__':
    main()
