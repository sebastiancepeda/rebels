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
###############################
def load_data(ImageShape, PatternShape, winSize, n_image = 21):
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
    
    w_t = numpy.load('../data/w_t.npy')
    params_updater(w_t)
    print('Show test images... ')
    for i in numpy.arange(21, 41):
        print('Test image: {}'.format(i))
        X_train, y_train = load_data(ImageShape, PatternShape, winSize, i)
        y_preds  = output_model(X_train)
        output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
        aux_image = output_image
        img = numpy.floor(output_image*255)
        cv2.imwrite('debug/y_preds-'+str(i)+'.png',img)

if __name__ == '__main__':
    main()
