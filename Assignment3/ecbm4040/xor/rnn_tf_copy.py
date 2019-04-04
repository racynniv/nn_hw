#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import numpy as np
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"

class MyGRUCell(RNNCell):
    """
    Your own basic GRUCell implementation that is compatible with TensorFlow. To solve the compatibility issue, this
    class inherits TensorFlow RNNCell class.

    For reference, you can look at the TensorFlow GRUCell source code. If you're using Anaconda, it's located at
    anaconda_install_path/envs/your_virtual_environment_name/site-packages/tensorflow/python/ops/rnn_cell_impl.py

    So this is basically rewriting the TensorFlow GRUCell, but with your own language.
    """

    def __init__(self, num_units, activation=None):
        """
        Initialize a class instance.

        In this function, you need to do the following:

        1. Store the input parameters and calculate other ones that you think necessary.

        2. Initialize some trainable variables which will be used during the calculation.

        :param num_units: The number of units in the GRU cell.
        :param activation: The activation used in the inner states. By default we use tanh.

        There are biases used in other gates, but since TensorFlow doesn't have them, we don't implement them either.
        """
        super(MyGRUCell, self).__init__(_reuse=None)
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        
        _BIAS_VARIABLE_NAME = "bias"
        _WEIGHTS_VARIABLE_NAME = "kernel"
        
        # cite:https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07
        
        # set number of units and activations
        self.num_units = num_units
        self.activation = activation or tf.tanh
        

    # The following 2 properties are required when defining a TensorFlow RNNCell.
    @property
    def state_size(self):
        """
        Overrides parent class method. Returns the state size of of the cell.

        state size = num_units

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return self.num_units

    @property
    def output_size(self):
        """
        Overrides parent class method. Returns the output size of the cell.

        :return: An integer.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        return self.num_units
    
    def build(self, input_shapes):
        input_depth = 1
        self.Wgate = self.add_variable(
            "gates/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self.num_units, 2 * self.num_units],
            initializer=None)
        self.bgate = self.add_variable(
            "gates/%s" % _BIAS_VARIABLE_NAME,
            shape=[2 * self.num_units],
            initializer=(init_ops.constant_initializer(1.0, dtype=self.dtype)))
        self.Wcan = self.add_variable(
            "candidate/%s" % _WEIGHTS_VARIABLE_NAME,
            shape=[input_depth + self.num_units, self.num_units],
            initializer=None)
        self.bcan = self.add_variable(
            "candidate/%s" % _BIAS_VARIABLE_NAME,
            shape=[self.num_units],
            initializer=(init_ops.zeros_initializer(dtype=self.dtype)))

        self.built = True

    def call(self, inputs, state):
        """
        Run one time step of the cell. That is, given the current inputs and the state from the last time step,
        calculate the current state and cell output.

        You will notice that TensorFlow GRUCell has a lot of other features. But we will not try them. Focus on the
        very basic GRU functionality.

        Hint 1: If you try to figure out the tensor shapes, use print(a.get_shape()) to see the shape.

        Hint 2: In GRU there exist both matrix multiplication and element-wise multiplication. Try not to mix them.

        :param inputs: The input at the current time step. The last dimension of it should be 1.
        :param state:  The state value of the cell from the last time step. The state size can be found from function
                       state_size(self).
        :return: A tuple containing (new_state, new_state). For details check TensorFlow GRUCell class.
        """
        #############################################
        #           TODO: YOUR CODE HERE            #
        #############################################
        # catconate the state and inputs matrices (making a ?x65 matrix)
        gates = array_ops.concat([inputs, state], 1)
        # create the gate input (a [?x65]*[65*2*64] = ?x2*64 matrix)
        gate_i = math_ops.matmul(gates,self.Wgate)
        # the bias to all terms of the tensor (output ?x2*64 matrix)
        gate_i = nn_ops.bias_add(gate_i,self.bgate)
        # sigmoid of the gate input (output ?x2*64 matrix)
        v_out = math_ops.sigmoid(gate_i)
        # split into two parts, the reset and u values (each a ?x64 matrix)
        r, u = array_ops.split(value=v_out,num_or_size_splits=2,axis=1)
        # ham multiply the reset value by the state value (output ?x64 matrix)
        r_state = r*state
        # create a new array of the concat input (?x1) and the r_state (?x64) making a ?x65
        concat = array_ops.concat([inputs,r_state],1)
        # matmultiply the new array with the weight tensor (?x65 * 65x64 = ?x64)
        can = math_ops.matmul(concat,self.Wcan)
        # add bias (?x64)
        can = nn_ops.bias_add(can,self.bcan)
        # use activation function to get the activated can (?x64)
        c = self.activation(can)
        # create the new output state (?x64 matrix)
        new_state = u*state+(1-u)*c
        
        return(new_state,new_state)