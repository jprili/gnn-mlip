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
    def __init__(self, layers: list, r_c, r_s, eta1, eta2, lbd, zeta):
        super().__init__()

        # properties of the subnet
        self._input  = Input(shape = (2, ))
        self.layers  = layers
        self._output = Dense(1, activation = "softmax")
        
        self.r_c  = r_c   # angstrom
        self.r_s  = r_s 
        self.eta1 = eta1
        self.eta2 = eta2
        self.lbd  = 1 if lbd == 1 else -1
        self.zeta = zeta

    def _norm(self, r):
        """
        macro for 2-d norm
        """
        return np.sqrt(np.dot(r, r))

    def _cut_off(self, r: float) -> float:
        """
        radial cutoff function
        """
        out = 0
        if r <= self.r_c:
            out = 0.5 * np.cos(np.pi * (r/self.r_c) + 1)
    
        return out

    def _trim_padding(self, inputs):
        mask = tf.reduce_any(inputs != -99, axis=-1)
        mask = tf.cast(mask, dtype=inputs.dtype)
               
        return inputs * tf.expand_dims(mask, axis=-1)

    def call(self, inputs, training = False):
        """
        Feed-forward algorithm for the model.
        calculate the values of the symmetry functions
        for each atom and applies to the same layers.
        """
        
        inputs = self._trim_padding(inputs)
        e_curr = 0
        g1s = self.get_g1is(inputs)
        g2s = self.get_g2is(inputs)

        for idx, in_vec in enumerate(zip(g1s, g2s)):
            in_vec = np.array(in_vec)
            x = self._input(in_vec)
            
            for layer in self.layers:
                # dropout only applies on training
                # for reqularisation
                if isinstance(layer, Dropout):
                    x = layer(x, training = training)
                    
                x = layer(x)

            e_curr += self._output(x).output

            # sum of all the atomic energy contributions
            # non trainable (just a linear sum)
            e_total = Dense(1, trainable = False)(e_curr)
        return e_total