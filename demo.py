"""
This is the main file for running the experiments found in the paper.
Specifically, the following cases are evaluated for memory usage:
(a) Synthetic dataset containing r repeats of each row
(b) Synthetic dataset containing r repeats of each row plus random noise
(c) Real-world dataset of features extracted from URLs
In addition, performance of scalar arithmetic operations is tested.
"""

import argparse
import numpy as np
from scipy import sparse
from datetime import datetime

from src.delta import delta_csr_matrix
from src.util import *

def synthetic_data_test(M, N, R, block_size, n_samples):
    """
    Test the memory savings of delta encoding using a synthetic dataset that has
    been constructed to contain repeated rows. Tunable parameters include:
    M: Number of rows in the dataset
    N: Number of columns in the dataset
    R: Number of times each row is repeated. Must cleanly divide M.
    """
    print("[%s] Starting basic synthetic data test..." % datetime.now().isoformat())
    print("Parameters: %d x %d matrix with repetition factor %d" % (M, N, R))
    dataset_chunk = sparse.random(M // R, N, format='csr')
    # achieve repetition by combining R copies of the chunk into a single matrix
    dataset = sparse.vstack([dataset_chunk for i in range(R)], format='csr')
    print("[%s] Memory usage of CSR matrix is %d bytes" % (datetime.now().isoformat(), csr_memory_usage(dataset)))
    print("[%s] Converting CSR matrix to delta CSR..." % datetime.now().isoformat())
    dataset_delta = delta_csr_matrix(dataset, block_size=block_size, n_samples=n_samples)
    print("[%s] Memory usage of delta CSR matrix is %d bytes" % (datetime.now().isoformat(), delta_csr_memory_usage(dataset_delta)))

def synthetic_data_test_noisy(M, N, R, block_size, n_samples, noise_level):
    """
    Same as the synthetic dataset test, except some noise is added to the
    repeated row copies to make them not quite identical. This tests the
    robustness of the delta encoding scheme to rows that are similar but not
    identical to each other. The amount of noise is controlled by the additional
    parameter noise_level.
    """
    print("[%s] Starting synthetic data test..." % datetime.now().isoformat())
    print("Parameters: %d x %d matrix with repetition factor %d" % (M, N, R))
    dataset_chunk = sparse.random(M // R, N, format='csr')
    chunks = [dataset_chunk]
    for i in range(R-1):
        noise = sparse.random(M // R, N, noise_level, format='csr')
        chunks.append(dataset_chunk + noise)
    dataset = sparse.vstack(chunks, format='csr')
    print("[%s] Memory usage of CSR matrix is %d bytes" % (datetime.now().isoformat(), csr_memory_usage(dataset)))
    print("[%s] Converting CSR matrix to delta CSR..." % datetime.now().isoformat())
    dataset_delta = delta_csr_matrix(dataset, block_size=block_size, n_samples=n_samples)
    print("[%s] Memory usage of delta CSR matrix is %d bytes" % (datetime.now().isoformat(), delta_csr_memory_usage(dataset_delta)))

def main():
    parser = argparse.ArgumentParser(description="Run delta CSR matrix demo workloads")
    parser.add_argument("-t", "--test", nargs="+", choices=["synthetic", "synthetic-noisy"],
                        help="specify which tests to run (if omitted, all tests are run)")
    args = parser.parse_args()
    if not args.test:
        # run all tests
        synthetic_data_test(1000000, 75000, 5, 5000, 500)
        synthetic_data_test_noisy(1000000, 75000, 5, 5000, 500, 0.001)
    else:
        # run only the tests specified by the user
        if "synthetic" in args.test:
            synthetic_data_test(1000000, 75000, 5, 5000, 500)
        if "synthetic-noisy" in args.test:
            synthetic_data_test_noisy(1000000, 75000, 5, 5000, 500, 1e-4)

if __name__ == '__main__':
    main()