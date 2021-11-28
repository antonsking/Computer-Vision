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
        
    np.random.shuffle(im_swp)
    np.random.shuffle(oh_lab_swp)
    
    mini_batch_x = np.array(np.array_split(im_swp,batch_size)).reshape((-1,196,32))
    mini_batch_y = np.array(np.array_split(oh_lab_swp,batch_size)).reshape((-1,10,32))
    
    return mini_batch_x,mini_batch_y


def fc(x, w, b):
    y = w @ x + b
    return y


def fc_backward(dl_dy, x, w, b, y):
    # TO DO
    return dl_dx, dl_dw, dl_db


def loss_euclidean(y_tilde, y):
    # TO DO
    return l, dl_dy

def loss_cross_entropy_softmax(x, y):
    # TO DO
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
    # TO DO
    return w, b

def train_slp(mini_batch_x, mini_batch_y):
    # TO DO
    return w, b

def train_mlp(mini_batch_x, mini_batch_y):
    # TO DO
    return w1, b1, w2, b2


def train_cnn(mini_batch_x, mini_batch_y):
    # TO DO
    return w_conv, b_conv, w_fc, b_fc


if __name__ == '__main__':
    main.main_slp_linear()
    main.main_slp()
    main.main_mlp()
    main.main_cnn()



