#!/usr/bin/env/ python
# ECBM E4040 Fall 2018 Assignment 2
# This Python script contains various functions for layer construction.

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    (B, H, W, C) = x.shape # get shapes
    (NF, FH, FW, FC) = w.shape
    new_height = ((H-FH + 2 * pad) // stride) +1 # set new height based on stride length and padding
    new_width = ((W-FW + 2 * pad) // stride) +1 # set new width based on stride length and padding
    
    out = np.zeros((B, new_height, new_width, NF)) # create empty vector for new values
    
    for n in range(B): # for each value in the batch
        padded = np.pad(x[n,:,:,:], ((pad,pad),(pad,pad),(0,0)), 'constant') # pad the sample
        for f in range(NF): # for weight
            for h in range(new_height): # each pixel in the new height
                for c in range(new_width): # each pixel in the new width
                    row_start = h*stride # find stride start and end points
                    row_end = h*stride + FH
                    col_start = c*stride
                    col_end = c*stride + FW
                    
                    # set out as the padded band being sent to the pixel plus the bias
                    out[n,h,c,f] = np.sum(padded[row_start:row_end,col_start:col_end,:] * w[f,:,:,:]) + b[f]
    return out


def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you +10 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (num_of_filters, filter_height, filter_width, channels).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    (B, new_height, new_width, NF) = d_top.shape # get shape values that matter
    (_, H, W, C) = x.shape
    (_, FH, FW, _) = w.shape
    
    dx = np.zeros(x.shape)
    dw = np.zeros(w.shape)
    db = np.zeros(b.shape)
    
    for n in range(B): # for each sample in the batch
        d_padded = np.pad(dx[n,:,:,:], ((pad,pad),(pad,pad),(0,0)),'constant') # set the padded derivative
        padded = np.pad(x[n,:,:,:], ((pad,pad),(pad,pad),(0,0)),'constant') # and the padded balues
        for f in range(NF):
            for h in range(new_height): # for each pixel
                for c in range(new_width):
                    row_start = h*stride # find the start and end points
                    row_end = h*stride + FH
                    col_start = c*stride
                    col_end = c*stride + FW
                    
                    # the derivative is just the weight times the upper derivative
                    d_padded[row_start:row_end,col_start:col_end,:] += w[f,:,:,:] * d_top[n,new_height-1,new_width-1,f]
                    # dw is the padded values times the above derivative
                    dw[n,:,:,:] += padded[row_start:row_end,col_start:col_end,:] * d_top[n,new_height-1,new_width-1,f]
                    # db is just based on the upper derivative
                    db[n] += d_top[n,new_height-1,new_width-1,f]
        # reshape the derivative to the dx shape to get dx
        dx[n,:,:,:] = d_padded[1:-1,1:-1,:]
        
    return dw,db


def max_pool_forward(x, pool_size, stride):
    """
    A Numpy implementation of 2-D image max pooling.

    Inputs:
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform max operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    (B,H,W,C) = x.shape # get shapes and sizes
    
    new_height = 1 + (H-pool_size) / stride
    new_width = 1 + (W-pool_size) / stride
    new_height = int(new_height)
    new_width = int(new_width)
    
    out = np.zeros((B,new_height,new_width,C)) # set output array
    
    for n in range(B): # for each sample in the batch
        for h in range(new_height): # for each pixel
            for w in range(new_width):
                row_start = h*stride # find the start and end values
                row_end = h*stride + pool_size
                col_start = w*stride
                col_end = w*stride + pool_size
                
                # the portion of the x frame taken is found
                part = x[n,row_start:row_end,col_start:col_end,:]
                # the out value for the specific pixel is the max of the pool previous (hence max pooling)
                out[n,h,w,:] = np.max(part.reshape((C,pool_size**2)),axis=1)
                
    return out

def max_pool_backward(dout, x, pool_size, stride):
    """
    (Optional, but if you solve it correctly, we give you +10 points for this assignment.)
    A Numpy implementation of 2-D image max pooling back-propagation.

    Inputs:
    :params dout: The derivatives of values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform max operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    
    :return dx: The derivative with respect to x
    You may find this website helpful:
    https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
    """
    #######################################################################
    #                                                                     #
    #                                                                     #
    #                         TODO: YOUR CODE HERE                        #
    #                                                                     #
    #                                                                     #
    #######################################################################
    (B,H,W,C) = x.shape # get important shapes
    (_, new_height, new_width,NF) = dout.shape
    
    dx = np.zeros(x.shape) # set dx as an empty array
    
    for n in range(B): # for each sample in the batch
        for c in range(C): # for each pixel in each channel
            for h in range(new_height):
                for w in range(new_width):
                    row_start = h*stride # find the window of operation
                    row_end = h*stride + pool_size
                    col_start = w*stride
                    col_end = w*stride + pool_size
                    part = x[n,row_start:row_end,col_start:col_end,c]
                    
                    # reshape the window to match the pool square
                    part = np.reshape(part,pool_size**2)
                    e_part = np.zeros(part.shape) # create an empty array
                    e_part[np.argmax(part)] = 1 # set the max value within the old window to 1
                    
                    # set dx as the reshaped binary matrix found above
                    dx[n,row_start:row_end,col_start:col_end,c] = np.reshape(e_part,(pool_size,pool_size)) * dout[n,h,w,c]
    return dx