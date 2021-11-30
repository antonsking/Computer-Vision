import cv2
import numpy as np
import os
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import math
import main_functions as main


def get_mini_batch(im_train, label_train, batch_size):
    ## For ease of batching
    im_swp = np.swapaxes(im_train,0,1)
    lab_swp = np.swapaxes(label_train,0,1)
    
    ## One hot encoding
    oh_lab_swp = []
    for lab in lab_swp:
        oh = np.zeros(10)
        oh[lab] = 1
        oh_lab_swp.append(oh)
    
    seed = 527
    np.random.seed(seed)
    np.random.shuffle(im_swp)
    np.random.seed(seed)
    np.random.shuffle(oh_lab_swp)
    
    mini_batch_x = np.array(np.array_split(im_swp,batch_size))#.reshape((-1,196,32))
    mini_batch_y = np.array(np.array_split(oh_lab_swp,batch_size))#.reshape((-1,10,32))
    
    return mini_batch_x,mini_batch_y


def fc(x, w, b):
    if len(x.shape) > 1:
        x = x.flatten()
    y = np.dot(w,x) + b
    
    return y


def fc_backward(dl_dy, x, w, b, y):
    dl_dx = np.dot(w.T,dl_dy)
    dl_dw = np.outer(dl_dy,x)
    dl_db = dl_dy

    return dl_dx, dl_dw, dl_db


def loss_euclidian(y_tilde, y):
    l = np.linalg.norm(y_tilde-y)**2
    dl_dy = 2*(y_tilde-y)
    
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x))
    
    l = -1 * (np.sum(y*np.log(1e-9 + softmax(x))) / len(y))
    dl_dy = softmax(x) - y
    
    return l, dl_dy

def relu(x):
    return np.maximum(x*1e-2,x)


def relu_backward(dl_dy, x, y):
    dl_dx = (x > 0).astype(int)
    return np.where(dl_dx==0., 0.01, dl_dx)


def conv(x, w_conv, b_conv):
    x = np.array([np.pad(x.reshape((14,14)),1,'constant')])
    X = im2col(x,3,3,1)
    y = X.dot(w_conv.reshape(9,-1)) + b_conv.reshape(3)

    return y.reshape(14,14,-1)


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    ## Assuming x is result from conv layer 
    dl_db = np.sum(dl_dy.reshape(-1,3),axis=0)
    dl_dw = dl_dy * x
    
    return dl_dw, dl_db

def pool2x2(x):
    out = [[0 for i in range(7)] for i in range(7)]
    wk = x.reshape((14,14))
    for i in range(0,14,2):
        for j in range(0,14,2):
            wnd = wk[i:i+2,j:j+2]
            out[i//2][j//2] = np.max(wnd)
            
    return np.array(out).reshape((7,7,1))

def pool2x2_backward(dl_dy, x, y):
    out = [[0 for i in range(14)] for i in range(14)]
    wk = x.reshape((14,14))
    for i in range(0,14,2):
        for j in range(0,14,2):
            wnd = wk[i:i+2,j:j+2]
            xx,yy = np.unravel_index(wnd.argmax(), wnd.shape)
            out[i+xx][j+yy] = np.max(wnd)
            
    return np.array(out)


def flattening(x):
    return np.ravel(x, order='F')


def flattening_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def train_slp_linear(mini_batch_x, mini_batch_y):
    batch_size,batches,size_img = mini_batch_x.shape
    n = 10
    m = size_img
    lr = 1e-4
    dr = 0.9
    w = np.random.normal(0, 1, size=(n,m))
    b = np.zeros((n))
    k = 0

    for itr in range(10000):
        if itr % 1000 == 0:
            lr *= dr

        dl_dw = 0
        dl_db = 0

        for i in range(batch_size):
            x = mini_batch_x[:,k][i]
            y = mini_batch_y[:,k][i]
            
            y_tilde = fc(x, w, b)
            l,dl_dy = loss_euclidian(y_tilde, y)
            dl_dx, dl_dw_n, dl_db_n = fc_backward(dl_dy, x, w, b, y)

            dl_dw += dl_dw_n
            dl_db += dl_db_n

        k += 1
        if k >= batches:
            k = 0

        w -= (lr * dl_dw)
        b -= (lr * dl_db)
        
    return w,b

def train_slp(mini_batch_x, mini_batch_y):
    batch_size,batches,size_img = mini_batch_x.shape
    n = 10
    m = size_img
    lr = 1e-3
    dr = 0.9
    w = np.random.normal(0, 0.5, size=(n,m))
    b = np.zeros((n))
    k = 0

    for itr in range(10000):
        if itr % 1000 == 0:
            lr *= dr

        dl_dw = 0
        dl_db = 0

        for i in range(batch_size):
            x = mini_batch_x[:,k][i]
            y = mini_batch_y[:,k][i]
            
            fc_out = fc(x, w, b)
            l,dl_dy = loss_cross_entropy_softmax(fc_out, y)
            dl_dx, dl_dw_n, dl_db_n = fc_backward(dl_dy, x, w, b, y)

            dl_dw += dl_dw_n
            dl_db += dl_db_n

        k += 1
        if k >= batches:
            k = 0

        w -= (lr * dl_dw)
        b -= (lr * dl_db)
        
    return w,b

def train_mlp(mini_batch_x, mini_batch_y):
    batch_size,batches,size_img = mini_batch_x.shape
    n = 10
    m = size_img
    lr = 1e-3
    dr = 0.9
    w1 = np.random.normal(0, 1, size=(30,196))
    w2 = np.random.normal(0, 1, size=(10,30))
    b1 = np.zeros((30))
    b2 = np.zeros((10))
    k = 0
    _ = None



    for itr in range(50000):

        if itr % 1000 == 0: 
            lr *= dr

        if itr % 10000 == 0:
            print(itr)

        dl_dw_1 = 0
        dl_dw_2 = 0
        dl_db_1 = 0
        dl_db_2 = 0

        for i in range(batch_size):
            x = mini_batch_x[:,k][i]
            y = mini_batch_y[:,k][i]
            
            ## Forward
            fc_1 = fc(x, w1, b1)
            relu_ = relu(fc_1)
            fc_2 = fc(relu_, w2, b2)
            l,dl_dy = loss_cross_entropy_softmax(fc_2, y)
            
            ## Backward
            dl_dx_2, dl_dw_n_2, dl_db_n_2 = fc_backward(dl_dy, relu_, w2, _, _)  
            dl_dx_relu = relu_backward(_, dl_dx_2, _)
            dl_dx_1, dl_dw_n_1, dl_db_n_1 = fc_backward(dl_dx_relu, x, w1, _, _)

            dl_dw_1 += dl_dw_n_1
            dl_db_1 += dl_db_n_1
            dl_dw_2 += dl_dw_n_2
            dl_db_2 += dl_db_n_2

        k += 1
        if k >= batches:
            k = 0

        w1 -= (lr * dl_dw_1)
        b1 -= (lr * dl_db_1)
        w2 -= (lr * dl_dw_2)
        b2 -= (lr * dl_db_2)

    return w1, b1, w2, b2

def train_cnn(mini_batch_x, mini_batch_y):
    batch_size,batches,size_img = mini_batch_x.shape
    n = 10
    m = size_img
    lr = 1e-3
    dr = 0.9
    h,w,c1,c2 = 14,14,1,3
    w_conv = np.random.normal(0,0.1,size=(3,3,1,3))
    b_conv = np.random.normal(0,0.1,size=(c2,1))
    w_fc = np.random.normal(0,0.1,size=(10,147))
    b_fc = np.random.normal(0,0.1,size=(10,1))
    k = 0
    _ = None

    st = t.time()

    for itr in range(5000):
        if itr % 1000 == 0: 
            lr *= dr

        if itr % 10000 == 0:
            end = t.time()
            print(end-st)
            st = t.time()
            print(itr)

        dl_dw_1 = 0
        dl_dw_2 = 0
        dl_db_1 = 0
        dl_db_2 = 0

        for i in range(batch_size):
            x = mini_batch_x[:,k][i]
            y = mini_batch_y[:,k][i]
            
            ## Forward
            conv_1 = conv(x, w_conv, b_conv)
            relu_1 = relu(conv_1)
            pooled = pool2x2(relu_1)
            flattened = flattening(pooled)
            fc_1 = fc(flattened, w_fc, b_fc)
            l,dl_dy = loss_cross_entropy_softmax(fc_1,y)
            
            ## Backward
            dl_dx_2, dl_dw_n_2, dl_db_n_2 = fc_backward(dl_dy, flattened, w_fc, b_fc, y)  
            dl_dx_1 = flattening_backward(dl_dy, pooled, y)
            pool_back = pool2x2_backward(dl_dy, relu_1, y)
            relu_back = relu_backward(dl_dy, conv_1, y)
            conv_back = conv_backward(dl_dy, x, w_conv, b_conv, y)
            
            
    ## TODO

        k += 1
        if k >= batches:
            k = 0

    ## TODO


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    #main.main_cnn()



