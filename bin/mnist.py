import cPickle, gzip, numpy
import theano
import theano.tensor as T
import lasagne
import scipy
import cv2

def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2, drop_hidden=.5):
    network = lasagne.layers.InputLayer(shape=(None, 784), input_var=input_var)
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
    network = lasagne.layers.DenseLayer(network, 10, nonlinearity=last_nonlin)
    return network

# Dataset
print('Loading dataset... ')
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Model
print('Creating model... ')
input_var = T.dmatrix('inputs')
target_var = T.ivector('targets')
network = build_custom_mlp(input_var, int(2), int(100), float(0.0), float(0.0))
prediction = lasagne.layers.get_output(network)
t2 = theano.tensor.extra_ops.to_one_hot(target_var, 10, dtype='int32')
error = lasagne.objectives.categorical_crossentropy(prediction, t2)
loss = error.mean()
params = lasagne.layers.get_all_params(network, trainable=True)
train_fn = theano.function([input_var, target_var], [loss,  error])
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

def func(params, *args):
        x = args[0]
        t = args[1]
        if(len(args) >= 3):
            p_data = args[2]
        params_updater(params)
        return train_fn(x,  t)[0]
    
def fprime(params, *args):
    x = args[0]
    t = args[1]
    if(len(args) >= 3):
        p_data = args[2]
    else:
        p_data = 1
    params_updater(params)
    return grad_giver(x, t, p_data)
    
# Main loop
print('Main loop... ')
w = params_giver()
args = (train_set[0], train_set[1].astype(numpy.int32))
m = 0
while(True):
    e = func(w, *args)
    g = fprime(w, *args)
    m = 0.99*m + 0.1*g#/numpy.abs(g).max()
    w = w - m
    print('e: {}'.format(e))
    print('numpy.abs(g).mean(): {}'.format(numpy.abs(g).mean()))
