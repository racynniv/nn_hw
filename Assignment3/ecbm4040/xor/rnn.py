#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops

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
        
        # cite:https://becominghuman.ai/understanding-tensorflow-source-code-rnn-cells-55464036fc07
        # cite:https://isaacchanghau.github.io/post/lstm-gru-formula/
        
        # set number of units and activations
        self.num_units = num_units
        self.activation = activation or tf.tanh
        self.Wz = self.add_variable("update_gate/W",shape = [64,64], initializer = None)
        self.Uz = self.add_variable("update_gate/U",shape = [1,64], initializer = None)
        self.Wr = self.add_variable("reset_gate/kernels",shape = [64,64], initializer = None)
        self.Ur = self.add_variable("reset_gate/bias",shape = [1,64], initializer = None)
        self.Wh = self.add_variable("can/W",shape = [64,64], initializer = None)
        self.Uh = self.add_variable("can/U",shape = [1,64], initializer = None)
        

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
        # find the value of the update gate
        zx = math_ops.matmul(inputs,self.Uz)
        zh = math_ops.matmul(state,self.Wz)
        z = math_ops.sigmoid(zx+zh)
        
        #find the value of the reset gate
        rz = math_ops.matmul(inputs,self.Ur)
        rh = math_ops.matmul(state,self.Wr)
        r = math_ops.sigmoid(rz+rh)
        
        # find candidate value
        hz = math_ops.matmul(inputs,self.Uh)
        hh = math_ops.matmul(r*state,self.Wh)
        h = self.activation(hz+hh)
        
        new_state = (1-z)*state + z*h
        
        return(new_state,new_state)