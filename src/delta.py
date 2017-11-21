"""
Modified CSR matrix format which delta-compresses similar rows to reduce
memory usage
"""

import types
import numpy as np

from scipy.sparse.compressed import _cs_matrix
from scipy.sparse.sputils import IndexMixin, isshape, get_index_dtype, getdtype, isintlike
from scipy.sparse.data import _data_matrix
from scipy.sparse import isspmatrix, isspmatrix_csr, csr_matrix

from .hash_similarity import HashSimilarityDetector
from .index_similarity import IndexSimilarityDetector
from .util import *

class delta_csr_matrix(csr_matrix, IndexMixin):
    """
    Compressed Sparse Row matrix with delta encoding

    This data structure behaves like the regular csr_matrix, but rows that are
    sufficiently similar to each other end up being represented as differences,
    or "deltas" from a reference row. As such, an additional data array is
    needed to specify which rows are "normal" rows and which are deltas, as
    well as which row is the reference for each delta.

    Like its normal counterpart, this can be instantiated as:
        delta_csr_matrix(D):
            from dense matrix or ndarray D
        
        delta_csr_matrix(S):
            from sparse matrix S

        delta_csr_matrix((M,N), [dtype]):
            empty MxN delta_csr_matrix with specified dtype (default 'd')

        delta_csr_matrix((data, (row_ind, column_ind)), [shape=(M,N)]):
            from raw COO data, such that a[row_ind[k], col_ind[k]] = data[k]

    But the "direct" instantiation differs from that of the regular CSR matrix
    due to the addition of the delta pointer array:
        delta_csr_matrix((data, indices, indptr, deltas), [shape=(M,N)]):
            the standard CSR representation modified to account for delta
            encoding: if row i is stored directly, this is indicated with
            deltas[i] = i, and the format is the same as standard CSR: the
            column indices are stored in indices[indptr[i]:indptr[i+1]] and the
            corresponding values are stored in data[indptr[i]:indptr[i+1]]. On
            the other hand, if row i is stored as a delta from some other row,
            then indices[indptr[i]:indptr[i+1]] and data[indptr[i]:indptr[i+1]]
            define the locations and values of nonzero elements of a delta
            vector, which should be added to the row specified by deltas[i]
            to get the true values for this row.
    """

    format = "dcsr"

    def __init__(self, arg1, block_size=None, n_samples=None, shape=None, dtype=None, copy=False):
        _data_matrix.__init__(self)

        # case 1: instantiate from another sparse matrix
        if isspmatrix(arg1):
            if arg1.format == self.format:
                self._set_self(arg1)
            elif arg1.format == "csr":
                # create a generator expression for iterating over rows of arg1
                row_gen = (arg1[i,:] for i in range(arg1.shape[0]))
                self._construct_from_iterable(row_gen, arg1.dtype,
                                              get_index_dtype(maxval=max(*arg1.shape)),
                                              block_size, n_samples, shape=arg1.shape,
                                              data_size=arg1.data.shape[0])
            else:
                raise NotImplementedError("Instantiation from sparse matrix not yet ready")

        # case 2: instantiate from some kind of raw data
        elif isinstance(arg1, tuple):
            if isshape(arg1):
                # input is size specification (M,N) for empty matrix
                # code mostly taken from scipy CSR implementation, other than an
                # additional line to instantiate deltas array
                self.shape = arg1
                M,N = self.shape
                idx_dtype = get_index_dtype(maxval=max(M,N))
                self.data = np.zeros(0, getdtype(dtype, default='float'))
                self.indices = np.zeros(0, idx_dtype)
                self.indptr = np.zeros(self._swap((M,N))[0] + 1, dtype=idx_dtype)
                self.deltas = np.zeros(0, dtype=idx_dtype)
            else:
                if len(arg1) == 2:
                    # COO data format
                    raise NotImplementedError("Instantiation from COO format not yet ready")
                elif len(arg1) == 3 or len(arg1) == 4:
                    # contents of the tuple are the raw data structures
                    (self.data, self.indices, self.indptr) = arg1[:3]
                    # use given shape or automatically infer one
                    if shape is not None:
                        self.shape = shape
                    else:
                        M = indptr.shape[0] - 1
                        N = np.max(indices)
                        self.shape = (M,N)
                    # a fourth array, for the deltas pointer, should always be
                    # given in general use, but we also allow for the case where
                    # it is omitted in order to maintain backwards compatibility
                    # with superclass methods. In this case we just let each
                    # deltas[i] = i; in other words treating this matrix as a
                    # standard CSR matrix with no delta encoding
                    self.deltas = arg1[3] if len(arg1) > 3 else np.arange(self.shape[0])

        # case 3: instantiate from generator object
        elif isinstance(arg1, types.GeneratorType):
            self._construct_from_iterable(arg1, getdtype(dtype, default='float'), 
                                          np.int32, block_size,
                                          n_samples, shape)

        # case 4: instantiate from dense matrix / array
        else:
            try:
                arg1 = np.asarray(arg1)
            except:
                raise ValueError("unrecognized delta_csr_matrix constructor usage")
            # create a generator expression for iterating over rows of arg1
            row_gen = (arg1[i,:] for i in range(arg1.shape[0]))
            self._construct_from_iterable(row_gen, arg1.dtype,
                                          get_index_dtype(maxval=max(*arg1.shape)),
                                          block_size, n_samples, shape=arg1.shape)

        self.check_format(full_check=False)

    def _set_self(self, other):
        self.data = other.data
        self.indices = other.indices
        self.indptr = other.indptr
        self.deltas = other.deltas
        self.shape = other.shape

    def _append_row_data(self, row_data, num_rows_added, data_added):
        """
        Adds values and index information for a new row onto the end of the
        component data structures. Agnostic to whether the given row data
        represents an actual row or a delta.

        Returns how much data was added to the data array
        """

        # extract the nonzero elements and their column indices
        if isspmatrix(row_data) and row_data.format == "csr":
            indices = row_data.indices
            data = row_data.data
        else:
            indices = np.nonzero(row_data)[0]
            data = row_data[indices]

        # update the data array and column indices with the values and locations
        # of the new row data. Since both arrays are dynamically allocated we
        # may have to grow them first
        if data_added + data.shape[0] > self.data.shape[0]:
            self.data = np.resize(self.data, self.data.shape[0] * 2)
            self.indices = np.resize(self.indices, self.indices.shape[0] * 2)
        # append the nonzero elements of the new row to data
        self.data[data_added:data_added+data.shape[0]] = data
        # do the same with their column indices
        self.indices[data_added:data_added+data.shape[0]] = indices

        # Update indptr; the new row goes up to one past the end of the
        # updated data array
        # Since indptr is dynamically allocated we must allocate more space
        # if needed
        if self.indptr.shape[0] <= num_rows_added + 1:
            self.indptr.resize(self.indptr.shape[0] * 2)
        self.indptr[num_rows_added+1] = data_added + data.shape[0]

        return data.shape[0]

    def _construct_from_iterable(self, rows, dtype, idx_dtype, block_size=None, n_samples=None, shape=None, data_size=10000):
        """
        Build a delta encoded sparse matrix row-by-row from an iterable of
        rows (e.g. a dense matrix or a CSR matrix)
        """

        # data structures are initially empty
        self.data = np.zeros(data_size, getdtype(dtype, default='float'))
        self.indices = np.zeros(data_size, idx_dtype)
        self.indptr = np.zeros(10, dtype=idx_dtype)
        self.deltas = np.zeros(10, dtype=idx_dtype)

        # keep track of which rows have been used as reference rows already
        reference_rows = {}

        M, N = shape if shape is not None else (None, None)

        # keep track of how many rows we have added thus far
        num_rows_added = 0
        # keep track of how large the data array currently is
        data_added = 0

        # use a HashSimilarityDetector to locate reference rows
        if block_size is None:
            block_size = N // 10
        sd = HashSimilarityDetector(block_size, n_samples)
        
        for row in rows:
            # from the first row we can infer the number of columns
            if N is None:
                N = row.shape[-1]
            # all rows should be the same length
            if row.shape[-1] != N:
                raise ValueError("Inconsistent row sizes passed (expected %d, got %d)" % (N, row.shape[-1]))
            # before we start: we will have to update self.deltas at some point,
            # which being dynamically allocated may not have enough allocated
            # space to update. As such, we should expand it if necessary
            if self.deltas.shape[0] < num_rows_added + 1:
                self.deltas.resize(self.deltas.shape[0] * 2)

            # use the HashSimilarityDetector to locate a row sufficiently
            # similar to this one to serve as a reference row
            # first, represent the row as a string:
            row_str = vec_to_str(row)
            # then look up a match
            ref = sd.get_best_match(row_str)

            # if no match was found, store the row directly
            if ref == -1:
                data_added += self._append_row_data(row, num_rows_added, data_added)
                # update self.deltas so that this row points to itself
                self.deltas[num_rows_added] = num_rows_added
                # since this row was added directly, we can consider it as a
                # candidate for a reference row in the future
                sd.add(row_str, num_rows_added)
            else:
                # reconstruct the reference row
                ref_row = np.zeros(N)
                start_idx = self.indptr[ref]
                end_idx = self.indptr[ref+1]
                ref_row[self.indices[start_idx:end_idx]] = self.data[start_idx:end_idx]
                # now compute the difference
                row_as_array = row.toarray().flatten() if isspmatrix(row) else row
                delta = row_as_array - ref_row
                # add the delta vector to the matrix
                data_added += self._append_row_data(delta, num_rows_added, data_added)
                # update self.deltas to point to the reference row
                self.deltas[num_rows_added] = ref
                del ref_row
            num_rows_added += 1

        # Once all rows have been added we can infer the height, if not given
        if M is None:
            M = num_rows_added
        # sanity check that the number of rows added equals what we were told
        if num_rows_added != M:
            raise ValueError("Number of rows provided not consistent with specified shape")

        # update this object's shape variable
        self.shape = (M,N)

        # since we dynamically allocate our data structures for performance
        # reasons, we must resize them to reflect how much data has actually
        # been added.
        self.data = np.resize(self.data, data_added)
        self.indices = np.resize(self.indices, data_added)
        self.indptr.resize(num_rows_added+1)
        self.deltas.resize(num_rows_added)

    def __getitem__(self, key):
        """
        Matrix slicing and element access operator

        For the most part, we can piggyback off the superclass implementation,
        but we need to implement additional logic to deal with the case where
        a delta vector's reference row is not in the slice. In this case we
        must reconstruct the full row and store it directly.
        """

        row, col = self._unpack_index(key)

        # fast path for row optimized methods
        if isintlike(row):
            if isinstance(col, slice):
                return self._get_row_slice(row, col)

        # calling the superclass implementation will give us a CSR matrix in
        # which some of the rows might be delta encodings
        raw_matrix = super().__getitem__(key)

        # find out which rows from the matrix are included in the output
        if type(key[0]) == slice:
            start = key[0].start if key[0].start is not None else 0
            stop = key[0].stop if key[0].stop is not None else self.shape[0]
            step = key[0].step if key[0].step is not None else 1
            out_rows = range(start, stop, step)
        else:
            start = key[0]
            out_rows = [key[0]]

        # now correct each of the included rows
        for row in out_rows:
            pass

        return raw_matrix

    def _get_row_slice(self, i, cslice):
        """
        Row slicing operator

        Can be very straightforwardly implemented as at most two calls to the
        superclass implementation
        """

        raw_slice_data = super()._get_row_slice(i, cslice)
        # if the sliced row happened to be directly stored, we are done
        if self.deltas[i] == i:
            return raw_slice_data
        # otherwise we must retrieve the reference row and add it to the
        # raw data, which represents a delta vector
        ref_data = super()._get_row_slice(self.deltas[i], cslice)

        raw_slice_csr = csr_matrix((raw_slice_data.data, raw_slice_data.indices,
                                    raw_slice_data.indptr),
                                   shape=raw_slice_data.shape)
        ref_csr = csr_matrix((ref_data.data, ref_data.indices, ref_data.indptr),
                             shape=ref_data.shape)

        row = raw_slice_csr + ref_csr
        return delta_csr_matrix((row.data, row.indices, row.indptr, np.zeros(1)),
                                shape=row.shape)

    def _with_data(self, data, copy=True):
        """
        Returns a version of this array where the data array is replaced with
        the given one. By default the other data structures are copied.
        """
        if copy:
            return delta_csr_matrix((data, self.indices.copy(),
                                     self.indptr.copy(), self.deltas.copy()),
                                    shape=self.shape, dtype=self.dtype)
        else:
            return delta_csr_matrix((data, self.indices, self.indptr, self.deltas),
                                    shape=self.shape, dtype=self.dtype)

    def getrow(self, i):
        """
        Access an individual row in this sparse matrix.
        """

        # right now we implement this simply using slicing
        return self._get_row_slice(i, slice(None))

    def toarray(self, order=None, out=None):
        """
        Convert this sparse matrix into a dense Numpy ndarray. We can mostly
        rely on the CSR code to do the heavy lifting but need to correct for
        delta encoded rows after the fact.
        """

        result = super().toarray(order, out)
        # the dense ndarray we got out of the super call is not necessarily valid,
        # as it may still contain delta vectors. We now check each row, and if
        # a row is a delta vector we add the reference row to it to restore its
        # original data
        for row in range(self.shape[0]):
            if self.deltas[row] != row:
                result[row, :] += result[self.deltas[row], :]

        return result
