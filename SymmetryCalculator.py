import os 
import os.path as path
import numpy as np
import csv
from pathlib import Path

class SymmetryCalculator:
    def __init__(self, dir = "./dat/data-set-2016-TiO2", r_c = 6.5):
        self.dir = Path(dir)
        
        self.is_periodic = True # no non-periodic files in default set
        self.start_idx   = 0
        self.positions   = np.array([])
        self.unit_size   = 0
        self.prim_vecs   = np.zeros((3, 3))
        self.combs       = np.array([])
        self.symmetries  = np.array([])
        self.r_c         = r_c

        self.s_no        = 0

        self.get_combinations()

    def get_max_unit_size(self):
        return 95
    
    def get_num_structures(self):
        return 7815
    
    def _norm(self, r):
        """
        macro for 2-d norm
        """
        return np.sqrt(np.dot(r, r))

    def get_g1_params(self):
        """
        reduced table 1 Artrith, Urban (2016)
        """
        return np.array([
            0.003214,
            0.035711,
            0.071421,
            0.124987,
            0.214264,
            0.357106,
            0.714213,
            1.428426
        ])
    
    def get_g2_params(self):
        """
        reduced version of table 2 Artrith, Urban (2016)
        """
        return np.array([
            [0.028569, -1.0, 1.0], [0.028569, -1.0, 2.0], [0.028569, -1.0, 4.0],
            [0.028569,  1.0, 1.0], [0.028569,  1.0, 2.0], [0.028569,  1.0, 4.0]
        ])

    def get_combinations(self):
        """
        get integer combinations for a 3d crystal
        for periodic boundaries.
        """
        out = np.zeros((3**3 - 1, 3))
        places = np.array([-1, 0, 1])

        # not the best but ran only once
        idx = 0

        # iterate through all combinations
        for i in places:
            for j in places:
                for k in places:
                    if comb := [i, j, k] == [0, 0, 0]:
                        continue
                        
                    out[idx] = comb
                    idx += 1
        self.combs = out

    def _cut_off(self, r):
        """
        radial cutoff function
        """
        out = 0
        if r <= self.r_c:
            out = 0.5 * (
                np.cos(np.pi * (r/self.r_c))
                + 1
                )
    
        return out

    def set_periodic(self, is_periodic = True):
        self.is_periodic = is_periodic

    def parse_xsf(self, dataset_path, s_no):
        file = path.join(
            dataset_path,
            f"structure{str(s_no).zfill(4)}.xsf"
        )

        with open(file, "r") as f:
            lines = f.readlines()
            
            self._set_unit_cell(lines)
            self._set_prim_vecs(lines)

    def _set_prim_vecs(self, lines):
        start_idx_ = 4
        end_idx_   = 6
        idxs = range(start_idx_, end_idx_ + 1)

        self.prim_vecs = np.array([
            np.array(lines[idx].split()).astype("float")
            for idx in idxs
        ])

    def _set_unit_cell(self, lines):
        
        # TODO: add periodic condition here
        
        size_info_idx_ = 8
        start_idx_     = 9
        dim_           = 3
    
        # get number of atoms in structure
        self.unit_size = int(lines[size_info_idx_].split(" ")[0])
        
        # initialise position array
        # cubed to accomodate for periodic condition
        positions = np.zeros((self.unit_size * 27, dim_))
        
        for idx, l in enumerate(lines[start_idx_ : start_idx_ + self.unit_size]):
            positions[idx, :] = np.array(
                l.split()[1 : 1 + dim_]
            ).astype("float")

        # should have unit_size non-zeros
        self.positions = positions

    def _construct_periodic(self):
        """
        extends position to periodic bounds
        """
        for c_idx, comb in enumerate(self.combs):
            d_vec = np.dot(self.prim_vecs, comb) # vector update
            for n_idx, pos in enumerate(self.positions[:self.unit_size]):
                idx = (c_idx + 1) * self.unit_size + n_idx
                self.positions[idx] = pos + \
                      d_vec

    def get_g1i(self, r_i, r, eta1):
        """
        symmetry function for two atoms, mu = 1
        r_s = 0
        """
        g1i_ptl = 0
        for r_j in r:
            r_ij = self._norm(r_i - r_j)

            # equation 4 in Behler, Parinello (2007)
            g1i_ptl += np.exp(-eta1 * (r_ij)**2) \
                     * self._cut_off(r_ij)
            
        return 2 * g1i_ptl

    def get_g2i(self, r_i, r, eta2, lbd, zeta):
        """
        symmetry function for three atoms, mu = 2
        """
        g2i_ptl = 0
        for idx, r_j in enumerate(r):
            for r_k in r[idx:]:
                r_ij = self._norm(r_i - r_j)
                r_ik = self._norm(r_i - r_k)
                r_jk = self._norm(r_j - r_k)

                # equation 5 in Behler, Parinello (2007)
                cutoff = self._cut_off(r_ij) \
                       * self._cut_off(r_ik) \
                       * self._cut_off(r_jk)
                
                if cutoff == 0.0:
                    continue
                else:
                    # t_ijk is a big time sink
                    cos_t_ijk = np.dot(r_ij, r_ik) / \
                        (self._norm(r_ij) * self._norm(r_ik))

                    g2i_ptl += (1 + lbd * cos_t_ijk)** zeta * \
                            np.exp(-eta2
                                    * (
                                        r_ij**2 + r_ik**2 + r_jk**2
                                        )) * cutoff

        return 2**(1 - eta2) * g2i_ptl

    def write_symmetry(self, dir_out, s_no):
        """
        writes symmetry of structure into a file
        """
        file = path.join(
            dir_out,
            f"sym{str(s_no).zfill(4)}.csv"
        )

        n_cols = 14
        out_shape = (self.get_max_unit_size(), n_cols)

        with open(file, "w") as f:
            self.parse_xsf(self.dir, s_no)

            if self.is_periodic:
                self._construct_periodic()

            writer = csv.writer(f)
            writer.writerow(np.arange(0, n_cols))

            out = np.zeros(out_shape)
            # iterate through atoms in unit cell
            for idx, pos in enumerate(self.positions[:self.unit_size]):
                r_js = self.positions[self.positions != pos].reshape(-1, 3)

                # params are lists, g2is 
                g1is = self.get_g1i(pos, r_js, self.get_g1_params())

                g2_params = self.get_g2_params()
                g2is = np.zeros(np.shape(g2_params)[0])
                for jdx, params in enumerate(g2_params):
                    g2is[jdx] = self.get_g2i(pos, r_js, *params)

                out[idx] = np.array([*g1is, *g2is])

            writer.writerows(out)
    
    def write_symmetries(self, dir_out, n_start, n_stop):
        for s_no in range(n_start, n_stop):
            self.write_symmetry(dir_out, s_no)
            print(f"structure {s_no} done!")


if __name__ == "__main__":
    sym_calc = SymmetryCalculator()
    n_start = 4
    n_stop  = 5
    # n_stop  = sym_calc.get_num_structures() + 1
    sym_calc.write_symmetries(Path(r"./dat/symmetries"), n_start, n_stop)