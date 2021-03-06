import unittest
import numpy as np
from scipy import sparse

from src.delta import delta_csr_matrix

M = 1000
N = 500
FP_TOLERANCE = 1e-10

class CorrectnessTests(unittest.TestCase):

    def setUp(self):
        block = sparse.random(M // 2, N, format='csr')
        self.csr = sparse.vstack([block, block], format='csr')
        self.dense = self.csr.toarray()
        self.delta_csr = delta_csr_matrix(self.dense)

    def test_dense_to_delta_csr(self):
        self.assertTrue((self.delta_csr.toarray() == self.dense).all(),
                        msg="Arrays differ after conversion to and from delta CSR")

    def test_csr_to_delta_csr(self):
        delta_csr_from_csr = delta_csr_matrix(self.csr)
        self.assertTrue((delta_csr_from_csr.toarray() == self.dense).all(),
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

    def test_range_slicing(self):
        slice_begin = M // 2 - 2
        slice_end = M // 2 + 3
        dcsr_slice = self.delta_csr[slice_begin:slice_end, :]
        self.assertTrue((dcsr_slice.toarray() == self.dense[slice_begin:slice_end, :]).all(),
                        msg="Sliced array contains incorrect values")

    def test_row_means(self):
        self.assertTrue(((self.delta_csr.mean(axis=1).flatten() - self.dense.mean(axis=1)) < FP_TOLERANCE).all(),
                        msg="Row wise means do not match")

    def test_matrix_mean(self):
        self.assertAlmostEqual(self.delta_csr.mean(), self.dense.mean())

    def test_col_means(self):
        self.assertTrue(((self.delta_csr.mean(axis=0).flatten() - self.dense.mean(axis=0)) < FP_TOLERANCE).all(),
                        msg="Column wise means do not match")

unittest.main()
