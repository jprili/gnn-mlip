import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
# suppress tensorflow warnings
import tensorflow as tf
import keras

from keras.models import Sequential 
from keras import Input
from keras.layers import Dense, \
                         Dropout


class BPNN(keras.Model):
    """
    The Model discussed in the 2007 paper by
    Behler and Parinello.

    This class contains an adaptive model which is
    the subnet for one atom.

    The subnet takes all the atom coordinates,
    and outputs the atomic contribution of energy
    """
    def __init__(self, layers: list, num_syms = 14):
        super().__init__()

        # properties of the subnet
        self.subnet   = Sequential([
            Input(shape = (num_syms,)),
            *layers
        ])
        self.num_syms = num_syms

    def call(self, inputs, training = False):
        """
        Feed-forward algorithm for the model.
        split each row and feed to the subnet
        for each atom and applies to the same layers.
        """
        # splits the rows
        syms = tf.unstack(inputs, axis = 1)

        sym_e_contribution = []
        for sym in syms:
            subnet_out = self.subnet(sym, training = training)
            sym_e_contribution.append(subnet_out)

        # turns the list into another tensor
        sym_preproc = tf.stack(sym_e_contribution, axis = 1)

        # add all the values in the tensor
        return tf.reduce_sum(sym_preproc)