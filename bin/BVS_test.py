from __future__ import print_function
from __future__ import division
import sys
import random
import os
import time
import numpy
import theano
import theano.tensor as T
import lasagne
import scipy
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import glob
import csv
import sqlite3
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
        ValidationSet = utils.getValues(config["validation_set"])
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
    test_n = ValidationSet
    test_idx    = numpy.arange(test_n.size)
    accuracy = numpy.zeros(test_n.size,)
    for idx in test_idx:
        print('* Test image: {}'.format(idx))
        x_image,  t_image,  mask_image = utils.get_images(ImageShape, PatternShape, winSize, test_n[idx])
        print('* get_mean. ')
        x_mean = utils.get_mean(x_image,  winSize,  ImageShape[2],  ImageShape)
        print('* get_std. ')
        x_std = utils.get_std(x_image,  winSize,  ImageShape[2],  ImageShape,  x_mean)
        print('* get_predictions. ')
        y_preds = utils.get_predictions(x_image, ImageShape, PatternShape, winSize, output_model,  x_mean, x_std)
        output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
        t_image = t_image.astype(numpy.float_)/255
        mask_image = mask_image.astype(numpy.float_)/255
        error_image,  accuracy[idx] = utils.get_error_image(output_image, t_image, mask_image)
        print('Accuracy[{}]: {}'.format(test_n[idx], accuracy[idx]))
        error_image = numpy.floor(error_image*255)
        cv2.imwrite('debug/error_image-'+str(test_n[idx])+'.png',error_image)
        # Output of model
        output_image = numpy.floor(output_image*255)
        cv2.imwrite('debug/y_preds-'+str(test_n[idx])+'.png',output_image)

if __name__ == '__main__':
    main()
