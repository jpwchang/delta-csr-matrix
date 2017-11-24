"""
transform_census_data.py
Author: Jonathan Chang (jpc362@cornell.edu)

Loads the 1990 US Census Data dataset (available at 
https://archive.ics.uci.edu/ml/datasets/US+Census+Data+%281990%29) and
transforms it into a sparse representation by encoding each of the categorical
attributes with a one-hot encoding.

The output file will be in svmlight format, which stores nonzero entries per
row and can therefore easily be loaded into a CSR matrix. However, this format
requires a label for each row, though this dataset does not present a
classification task. We get around this by simply storing the label +1 for all
rows.
"""

import sys
import pandas as pd
import numpy as np

def get_header_sizes(dataset):
    """
    Build a dictionary specifying how many array indices will be needed for
    each header in the original datset.
    """
    header_sizes = {}
    for header in dataset.columns:
        if header == "caseid":
            # caseid is just an index and we should skip it
            continue
        maxval = np.max(dataset[header])
        # we assume that every categorical feature can take on values 0 to
        # maxval, inclusive. Thus we need maxval+1 indices
        header_sizes[header] = maxval + 1
    return header_sizes

def transform_dataset(dataset, out_path):
    # find out how many indices we will need for each categorical feature
    feature_sizes = get_header_sizes(dataset)
    features = list(feature_sizes.keys()) # impose an iteration order
    with open(out_path, 'w') as fp:
        for row in dataset.itertuples():
            accum = 0 # an accumulating offset for feature index
            row_strs = ["+1"]
            for feature in features:
                # get the original value of the categorical feature
                raw_value = getattr(row, feature)
                true_index = int(raw_value) + accum
                row_strs.append("%d:1" % true_index)
                accum += feature_sizes[feature]
            fp.write(' '.join(row_strs))
            fp.write('\n')

def main():
    if len(sys.argv) < 3:
        print("Usage: transform_census_data.py <in file> <out file>")
        return
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    # load the 1990 Census dataset in its original form as a pandas dataframe
    df = pd.read_csv(in_path, header=0)
    # transform dataset into svmlight format and write it out to a file
    transform_dataset(df, out_path)

if __name__ == '__main__':
    main()
