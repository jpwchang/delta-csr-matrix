"""
transform_url_data.py
Author: Jonathan Chang (jpc362@cornell.edu)

Loads the dataset of URLs from the GitHub project "Using Machine Learning to
Detect Malicious URLs" (https://github.com/faizann24/Using-machine-learning-to-detect-malicious-URLs),
extracts features from each URL, and saves the resulting matrix to disk in
SVMlight format
"""

import numpy as np
import pandas as pd
import sys
from tld import get_tld
from sklearn.feature_extraction.text import CountVectorizer

def extract_domain(url):
    if '?' in url:
        url = url[:url.index('?')]
    try:
        return get_tld(url, fix_protocol=True, as_object=True).domain
    except:
        return "UNKNOWN_DOMAIN"

def extract_tld(url):
    if '?' in url:
        url = url[:url.index('?')]
    try:
        return get_tld(url, fix_protocol=True, as_object=True).suffix
    except:
        return "UNKNOWN_TLD"

def extract_filename(url):
    if '?' in url:
        url = url[:url.index('?')]
    if len(url) == 0:
        return ("", "NO_EXTENSION")
    # strip trailing slash
    if url[-1] == '/':
        url = url[:-1]
    components = url.split('/')
    if len(components) == 1:
        return ("", "NO_EXTENSION")
    filename = components[-1]
    filename_parts = filename.split('.')
    if len(filename_parts) == 1:
        return (filename, "NO_EXTENSION")
    return ('.'.join(filename_parts[:-1]), filename_parts[-1])

def transform_dataset(dataset, out_path):
    domains = np.unique(dataset.url.apply(extract_domain).values)
    tlds = np.unique(dataset.url.apply(extract_tld))
    filenames_and_extensions = dataset.url.apply(extract_filename)
    filenames = [p[0] for p in filenames_and_extensions]
    extensions = np.unique([p[1] for p in filenames_and_extensions])

    # create dictionaries mapping from features to indices
    domain_dict = dict(zip(domains, range(len(domains))))
    tld_dict = dict(zip(tlds, range(len(tlds))))
    extension_dict = dict(zip(extensions, range(len(extensions))))
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5,5), max_features=100)
    vectorizer.fit(filenames)
    filename_dict = vectorizer.vocabulary_
    tokenize = vectorizer.build_analyzer()

    # now we can iterate over the dataset and convert each URL to a feature vector
    with open(out_path, 'w') as fp:
        for row in dataset.itertuples():
            # get the label for this row
            row_str = ['+1' if row.label=="bad" else '-1']
            # get the URL and extract features
            url = row.url
            domain = extract_domain(url)
            tld = extract_tld(url)
            filename, extension = extract_filename(url)
            has_params = ('?' in url)
            # convert to feature vector
            accum = 0
            if has_params:
                row_str.append("0:1")
            accum += 1
            row_str.append("%d:1" % (domain_dict[domain] + accum))
            accum += len(domain_dict)
            row_str.append("%d:1" % (tld_dict[tld] + accum))
            accum += len(tld_dict)
            row_str.append("%d:1" % (extension_dict[extension] + accum))
            accum += len(extension_dict)
            for token in tokenize(filename):
                if token in filename_dict:
                    row_str.append("%d:1" % (filename_dict[token] + accum))
            # combine the features into an SVNlight formatted row string
            fp.write(' '.join(row_str))
            fp.write('\n')

def main():
    if len(sys.argv) < 3:
        print("Usage: transform_url_data.py <in file> <out file>")
        return
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    # load the URL dataset into a pandas dataframe
    df = pd.read_csv(in_path, header=0)
    transform_dataset(df, out_path)

if __name__ == '__main__':
    main()