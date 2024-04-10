

class BPNN(keras.Model):
    """
    The Model discussed in the 2007 paper by
    Behler and Parinello.

    This class contains an adaptive model which is
    the subnet for one atom.

    The subnet takes all the atom coordinates,
    and outputs the atomic contribution of energy
    """
    def __init__(layers: list, c, r_s, eta, lbd, zeta,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        # properties of the subnet
        self.layers  = layers
        self._input  = Input(shape = (2, ))
        self._output = Dense(1, activation = "softmax")
        
        self.c    = c   # angstrom
        self.r_s  = r_s 
        self.eta  = eta
        self.lbd  = 1 if lbd == 1 else -1
        self.zeta = zeta

    def _norm(self, r):
        """
        macro for 2-d norm
        """
        return np.sqrt(np.dot(r, r))

    def _cut_off(self, r: float, c: float) -> float:
        """
        radial cutoff function
        """
        out = 0
        if r <= c:
            out = 0.5 * np.cos(np.pi * (r/c) + 1)
    
        return out
    
    def get_g1i(self, r_i, r) -> float:
        """
        symmetry function for two atoms, mu = 1
        """
        g1i_ptl = 0
        for r_j in r:
            r_ij = self._norm(r_i - r_j)

            # equation 4 in Behler, Parinello (2007)
            g1i_ptl += np.exp(-eta * (r_ij - self.r_s)**2) \
                     * _cut_off(r_ij, c)
            
        return 2 * g1i

    def get_g1is(self, r):
        """calculate g1i for all atoms"""
        g1is = np.zeros(np.shape(r)[0])
        for idx, r_i in enumerate(r[:-1]):
            r_js = r[r != r_i].reshape(-1, 3)
            g1is[idx] = self.get_g1i(r_i, r_js)

        return g1is

    def get_g2i(self, r_i, r):
        """
        symmetry function for three atoms, mu = 2
        """
        g2i_ptl = 0
        for jdx, r_j in enumerate(r[:-1]):
            for r_k in r[jdx:]:
                r_ij = self._norm(r_i - r_j)
                r_ik = self._norm(r_i - r_k)
                r_jk = self._norm(r_j - r_k)
                t_ijk = np.dot(r_ij, r_ik) / \
                        (self._norm(r_ij) * self._norm(r_ik))

                # equation 5 in Behler, Parinello (2007)
                g2i_ptl += (1 + (self.lbd) * np.cos(t_ijk))**self.zeta * \
                           np.exp(-self.eta
                                  * np.sum(
                                      [r**2 for r in [r_ij, r_ik, r_jk]
                                    ])) * \
                           self._cut_off(r_ij) * self._cut_off(r_ik) * \
                           self._cut_off(r_jk)

        return 2**(1 - eta) * g2i_ptl

    def get_g2is(self, r):
        """calculate g2i for all atoms"""
        g2is = np.zeros(np.shape(r)[0])
        for idx, r_i in enumerate(r[:-2]):
            r_js = r[r != r_i].reshape(-1, 3)
            g2is[idx] = self.get_g1i(r_i, r_js)
        return g2is

    def call(self, inputs, training = False):
        """
        Feed-forward algorithm for the model.
        calculate the values of the symmetry functions
        for each atom and applies to the same layers.
        """
        n_atoms = np.shape(inputs)[0]
        e_curr  = 0
        g1s = get_g1is(inputs)
        g2s = get_g2is(inputs)

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