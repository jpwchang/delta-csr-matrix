import unittest
import numpy as np
from scipy import sparse

from src.delta import delta_csr_matrix

M = 1000
N = 500
FP_TOLERANCE = 1e-5

class CorrectnessTests(unittest.TestCase):

    def setUp(self):
        self.csr = sparse.random(M, N, format='csr')
        self.dense = self.csr.toarray()
        self.delta_csr = delta_csr_matrix(self.dense)

    def test_dense_to_delta_csr(self):
        self.assertTrue((self.delta_csr.toarray() == self.dense).all(),
                        msg="Arrays differ after conversion to and from delta CSR")

    def test_csr_to_delta_csr(self):
        delta_csr_from_csr = delta_csr_matrix(self.csr)
        self.assertTrue((delta_csr_from_csr.toarray() == self.csr.toarray()).all(),
                        msg="Arrays differ after conversion from CSR")

    def test_row_slicing(self):
        for i in range(M):
            dense_row = self.dense[i,:]
            # first try row access using slicing notation
            dcsr_row_slice = self.delta_csr[i,:].toarray().flatten()
            self.assertTrue((dense_row == dcsr_row_slice).all(),
                            msg="Row %d does not match using slicing notation!" % i)
            # using getrow should yield same results
            dcsr_row_getrow = self.delta_csr.getrow(i).toarray().flatten()
            self.assertTrue((dense_row == dcsr_row_getrow).all(),
                            msg="Row %d does not match using getrow!" % i)

    def test_scalar_multiply(self):
        doubled_dense = 2 * self.dense
        doubled_dcsr = 2 * self.delta_csr
        self.assertTrue((abs(doubled_dcsr.toarray() - doubled_dense) < FP_TOLERANCE).all(),
                        msg="Scalar multiplication results differ between dense and delta CSR matrix!")

unittest.main()
