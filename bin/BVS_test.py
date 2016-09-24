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
    print('* Show test images... ')
    for i in numpy.arange(21, 41):
        print('Test image: {}'.format(i))
        x, t,  mask = utils.load_data(ImageShape, PatternShape, winSize, i)
        y_preds  = output_model(x)
        x = 0
        output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
        t = utils.reconstruct_image_3(t,w=winSize, PatternShape=PatternShape)
        mask = utils.reconstruct_image_3(mask,w=winSize, PatternShape=PatternShape)
        # Print accuracy and debug 3 color image
        error_image,  accuracy = utils.get_error_image(output_image, t, mask)
        print('Accuracy[{}]: {}'.format(i, accuracy))
        error_image = numpy.floor(error_image*255)
        cv2.imwrite('debug/error_image-'+str(i)+'.png',error_image)
        # Output of model
        output_image = numpy.floor(output_image*255)
        cv2.imwrite('debug/y_preds-'+str(i)+'.png',output_image)

if __name__ == '__main__':
    main()
