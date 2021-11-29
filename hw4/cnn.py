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

    l = -1 * (np.sum(y*np.log(softmax(x))) / len(y))
    dl_dy = softmax(x) - y
    
    return l, dl_dy

def relu(x):
    # TO DO
    return y


def relu_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def conv(x, w_conv, b_conv):
    # TO DO
    return y


def conv_backward(dl_dy, x, w_conv, b_conv, y):
    # TO DO
    return dl_dw, dl_db

def pool2x2(x):
    # TO DO
    return y

def pool2x2_backward(dl_dy, x, y):
    # TO DO
    return dl_dx


def flattening(x):
    # TO DO
    return y


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
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    #main.main_slp_linear()
    #main.main_slp()
    main.main_mlp()
    #main.main_cnn()



