import numpy as np
from scipy import sparse

def get_nonzero_indices(row):
    """
    Given a row from a matrix, return a list of indices in the row that have
    nonzero values
    """
    if sparse.isspmatrix_csr(row):
        return row.indices
    else:
        row = np.asarray(row)
        return np.nonzero(row)[0]

def overlap(a, b):
    """
    Given two arrays in sorted ascending order, compute the number of elements
    they have in common in linear time. Normalize this count by the number
    of elements in a and in b, and return the average of the two.
    """

    # fast path: if either array is empty there can be no overlap
    if a.shape[0] == 0 or b.shape[0] == 0:
        return 0

    a_idx, b_idx = 0, 0
    result = 0
    while a_idx < a.shape[0] and b_idx < b.shape[0]:
        # case 1: elements match
        if a[a_idx] == b[b_idx]:
            result += 1
            a_idx += 1
            b_idx += 1
        # case 2: element from a less than element from b
        elif a[a_idx] < b[b_idx]:
            # there could still be a match later in a, so don't advance b_idx
            a_idx += 1
        # case 3: element from b less than element from a
        else:
            # there could still be a match later in b, so don't advance a_idx
            b_idx += 1

    a_norm = result / a.shape[0]
    b_norm = result / b.shape[0]
    return (a_norm + b_norm) / 2

def vec_to_str(vec):
    result = np.empty(vec.shape[-1], dtype='object')
    result[:] = '0'
    if sparse.isspmatrix_csr(vec):
        nz = vec.indices
    else:
        nz = np.nonzero(vec)[0]
    result[nz] = '1'
    return ''.join(result)

def csr_memory_usage(X):
    """
    Estimate the total memory usage of the CSR matrix X
    """
    return X.data.nbytes + X.indices.nbytes + X.indptr.nbytes

def delta_csr_memory_usage(X):
    """
    Estimate the total memory usage of the delta encoded CSR matrix X
    """
    return X.data.nbytes + X.indices.nbytes + X.indptr.nbytes + X.deltas.nbytes