from __future__ import print_function
from __future__ import division
import sys
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
import utils

ImageShape      = (584,565,3)
PatternShape    = (584,565)
winSize             = (3, 3)
nbatch              = 100
alpha                = 1 # in [0.01,100]
beta                  = 0.1
gamma              = 10

def load_data():
    features 	   = Image.open('../DRIVE/training/images/21_training.tif','r')
    #features 	   = Image.open('../DRIVE/21a.gif','r')
    #features 	   = Image.open('../DRIVE/21d.bmp','r')
    #features 	 = Image.open('../DRIVE/AddBorder21.png','r')
    #features 	 = Image.open('../DRIVE/training/1st_manual/21_manual1.gif','r')
    
    features     = numpy.fromstring(features.tostring(), dtype='uint8', count=-1, sep='')
    features     = numpy.reshape(features,ImageShape,'A')
    
    image2     = Image.open('../DRIVE/training/1st_manual/21_manual1.gif','r')
    #image2     = Image.open('../DRIVE/training/mask/21_training_mask.gif','r')
    image2     = numpy.fromstring(image2.tostring(), dtype='uint8', count=-1, sep='')
    image2     = numpy.reshape(image2,PatternShape,'A')
    
    train_set  = utils.sliding_window(features, stepSize=1, w=winSize, dim=ImageShape[2],output=0)
    valid_set  = train_set
    test_set   = train_set
    
    train_set_t  = utils.sliding_window(image2, stepSize=1, w=winSize, dim=1,output=1)
    valid_set_t  = train_set_t
    test_set_t   = train_set_t
    
    #train_set,  train_set_t = balance(train_set,  train_set_t)
    
    print('train_set_t.mean()')
    print(train_set_t.mean())
    
    def shared_dataset(data_x, data_y, borrow=True):
        shared_x = numpy.asarray(data_x,dtype=theano.config.floatX)
        shared_y = numpy.asarray(numpy.squeeze(numpy.asarray(data_y)),dtype=theano.config.floatX)
        return shared_x, shared_y.astype(numpy.float64)

    test_set_x, test_set_y   = shared_dataset(test_set, test_set_t)
    valid_set_x, valid_set_y = shared_dataset(valid_set, valid_set_t)
    train_set_x, train_set_y = shared_dataset(train_set, train_set_t)
    
    return train_set_x, train_set_y, valid_set_x, valid_set_y,test_set_x, test_set_y

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    network = lasagne.layers.InputLayer(shape=(None, ImageShape[2], winSize[0], winSize[1]), input_var=input_var)
    if drop_input:
        network = lasagne.layers.dropout(network, p=drop_input)
    # Hidden layers and dropout:
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
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    input_var = T.dtensor4('inputs')
    target_var = T.dvector('targets')
    p = T.dvector('p')
    print("Building model and compiling functions...")
    if model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = build_custom_mlp(input_var, int(depth), int(width), float(drop_in), float(drop_hid))
    else:
        print("Unrecognized model type %r." % model)
        return
    prediction = lasagne.layers.get_output(network)
    error = prediction[:, 1] + target_var
    error = (error*error)
    loss = error.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = ((test_prediction[:, 1]-target_var)*(test_prediction[:, 1]-target_var)).mean()
    test_loss = test_loss.mean()
    a = numpy.zeros((2,2))
    a[0][0]     = 1
    a[0][1]     = 0
    a[1][0]     = 0
    a[1][1]     = alpha
    thresholded_prediction = T.dot(test_prediction,a)
    thresholded_prediction = T.argmax(test_prediction, axis=1)
    test_acc = T.mean(T.eq(thresholded_prediction, target_var),dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], [loss,  error])
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    output_model = theano.function([input_var], prediction)
    def params_giver():
        ws = []
        for w in params:
            params_fn = theano.function([],  w)
            w_out = params_fn()
            ws = numpy.append(ws,  w_out)
        return ws
    
    def grad_giver(x,  t,  p_data):
        gs = []
        for w in params:
            grad = T.grad(loss,  w)
            grad_fn = theano.function([input_var, target_var], grad)
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
        return train_fn(x,  t)
    
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
        return grad_giver(x, t, p_data)
        
    def callback(all_w):
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, nbatch, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1
        
        # Then we print the results for this epoch:
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
    '''
    Method that performs a plot of the error surface with respect to lamda
    '''
    def ghaph(args,  w_t,  g, start, end, points,  debug):
        phi = numpy.linspace(start, end, points)
        e_phi = numpy.linspace(0, 0, points)
        for k in numpy.arange(phi.size):
            e_phi[k] = func(w_t - phi[k]*g, *args)[0]
        if(debug >= 1):
            plt.plot(phi, e_phi)
            plt.show()
        return phi[numpy.argmin(e_phi)]
        
    def sampleData(training_data,  p_total,  n_samples = 10000):
        p_examples = utils.sliding_window2(p_total, stepSize=1, w=winSize, dim=1)
        p_examples = numpy.asarray(numpy.squeeze(numpy.asarray(p_examples)),dtype=theano.config.floatX).astype(numpy.float64)
        p_examples = p_examples/(1/1000000+p_examples.sum())
        x = training_data[0]
        y = training_data[1]
        data_out = numpy.zeros((x.shape[0],))
        p_dummy = numpy.zeros((n_samples,))+1
        inds = (-p_examples+numpy.random.rand(p_examples.shape[0], )*p_examples.mean()).argsort()
        x_out = x[inds[0:n_samples],:,:,:]
        y_out = y[inds[0:n_samples]]
        data_out[inds[0:n_samples]] = 1
        print("x_out.shape: {}".format(x_out.shape))
        print("y_out.shape: {}".format(y_out.shape))
        print('data_out.sum(): {}'.format(data_out.sum()))
        return (x_out, y_out, p_dummy,  data_out)
        #return (x, y, p_dummy,  data_out)
    
    def optimizer(func, x0, fprime, training_data, callback):
        print('Optimizer method. ')
        w_t = x0
        m_t = 0
        p = numpy.zeros((PatternShape[0],PatternShape[1]))+1/(ImageShape[0]*ImageShape[1])
        args = sampleData(training_data,  p)
        e_t = func(w_t , *args)
        print('deleting images... ')###########################################
        for f in glob.glob("debug/*/*.png"):
            os.remove(f)
        print('save input image... ')###########################################
        idx_x = X_val.shape[1]//2
        idx_y = X_val.shape[2]//2
        idx_z = X_val.shape[3]//2
        x_out = X_val[:, idx_x, idx_y, idx_z]
        x_image = numpy.pad(numpy.reshape(x_out,(PatternShape[0]-winSize[0]+1,PatternShape[1]-winSize[1]+1),'A'), (winSize[0]//2, winSize[1]//2), 'constant')
        x_image = numpy.floor(x_image)
        cv2.imwrite('debug/0-x_image.png',x_image)
        print('save target image... ')##########################################
        targ = numpy.pad(numpy.reshape(y_val,(PatternShape[0]-winSize[0]+1,PatternShape[1]-winSize[1]+1),'A'), (winSize[0]//2, winSize[1]//2), 'constant')
        t_image = numpy.floor(targ)
        cv2.imwrite('debug/0-t_image.png',t_image*255)
        for i in numpy.arange(num_epochs):            
            dedw = fprime(w_t , *args)
            g_t = dedw/(numpy.abs(dedw).max())
            #g_t = dedw
            m_t = 0.9*m_t + g_t*0.0001
            #lamda_t = ghaph(args,  w_t,  g_t, -1, 1, 10,  debug=0)
            lamda_t = 0.9
            w_t  = w_t + m_t*lamda_t
            e_t = func(w_t , *args)
            print('e_t[1].shape: {}'.format(e_t[1].shape))
            print('error_T image... ')
            #error_T = numpy.pad(numpy.reshape(e_t[1],(PatternShape[0]-winSize[0]+1,PatternShape[1]-winSize[1]+1),'A'), (winSize[0]//2, winSize[1]//2), 'constant')
            #cv2.imwrite('debug/error_t/{}-image.png'.format(i),error_T*255)
            args = sampleData(training_data, p)
            print("lamda_t: {}".format(lamda_t))
            print("numpy.abs(g_t).mean(): {}".format(numpy.abs(g_t).mean()))
            print("i: {}, e_t: {}".format(i, e_t[0]))
            print("error: {}".format(e_t[1].mean()))
            print("-----------------------------------------------------------")
            #if(numpy.abs(g_t).mean() < 1/1000000):
            #    break
            if(i % 1 == 0):
                #callback(w_t)
                print('save data image... ')
                data_out = args[3]
                print('data_out.shape: {}'.format(data_out.shape))
                im_data = numpy.pad(numpy.reshape(data_out,(PatternShape[0]-winSize[0]+1,PatternShape[1]-winSize[1]+1),'A'), (winSize[0]//2, winSize[1]//2), 'constant')
                cv2.imwrite('debug/data/{}-image.png'.format(i),im_data*255)
                print('save test image... ')
                y_out = output_model(X_val)
                y_out = numpy.pad(numpy.reshape(y_out,(PatternShape[0]-winSize[0]+1,PatternShape[1]-winSize[1]+1, 2),'A'), (winSize[0]//2, winSize[1]//2), 'constant')
                y_out = y_out[:,:,1:3]
                output_image = (y_out[:, :, 0] < y_out[:, :, 1]*alpha)*1
                img = numpy.floor(output_image*255)
                cv2.imwrite('debug/y_out/{}-image.png'.format(i),img)
                print('save error image... ')
                e_preds = y_out[:, :, 1] - targ
                e_img = e_preds*e_preds
                e_img = numpy.floor(e_img*255)
                cv2.imwrite('debug/error/{}-error_image.png'.format(i),e_img*255)
                print('save acc image... ')
                a = output_image*targ+(1-output_image)*(1-targ)
                cv2.imwrite('debug/acc/{}-acc_image.png'.format(i),a*255)
                p = a*p*1/1.1+(1-a)*p*1.1
                p_image = p
                p_image = p_image - p_image.min()
                p_image = p_image / p_image.max()
                cv2.imwrite('debug/p/{}-p_image.png'.format(i),numpy.floor(p_image*255))
    
    #optimizer(func, x0=params_giver(), fprime=fprime, training_data=(X_train,y_train),  callback=callback)
    scipy.optimize.fmin_l_bfgs_b(func, x0=params_giver(), fprime=fprime, args=(X_train, y_train), approx_grad=0, bounds=None, m=10, factr=10000000.0, pgtol=1e-05, epsilon=1e-08, iprint=0, maxfun=15000, maxiter=15000, disp=None, callback=callback)
    
    print('Show test image... ')
    y_preds   = [output_model(inputs) for inputs, targets in iterate_minibatches(X_val, y_val, 1, shuffle=False)]
    print('y_preds[0] :{}'.format(y_preds[0]))
    output_image = utils.reconstruct_image(y_preds,w=winSize, PatternShape=PatternShape, alpha=alpha)
    aux_image = output_image
    print('--------------Imagen resultante--------------')
    img = numpy.floor(output_image*255)
    cv2.imwrite('debug/image-last.png',img)
    print('aux_image.min(): {}'.format(aux_image.min()))
    print('aux_image.mean(): {}'.format(aux_image.mean()))
    print('aux_image.max(): {}'.format(aux_image.max()))
    
    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, nbatch, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print('--------------Final results--------------')
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
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
