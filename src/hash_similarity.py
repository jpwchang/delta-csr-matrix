import numpy as np
from collections import Counter

class HashSimilarityDetector():
    """
    Class that performs inexact matching of objects using hashes of fixed-size
    blocks. Based upon the technique of the same name first introduced in the
    paper "Difference Engine: Harnessing Memory Redundancy in Virtual Machines"
    by Gupta et al.

    In this implementation, objects are assumed to be represented as strings.
    """

    def __init__(self, block_size, random_samples=None, limit=None):
        self.table = {}
        self.block_size = block_size
        self.random_samples = random_samples
        self.limit = limit

    def add(self, new_obj, obj_identifier):
        """
        Add a new entry to the table of similarity candidates. The entry is
        identified using the provided identifier, which could be, for instance,
        an index into an array
        """
        # first, split the object into blocks
        blocks = [new_obj[i:i+self.block_size] for i in range(0, len(new_obj), self.block_size)]
        # now hash each block. Order matters (block '0000' at position 1 is not
        # the same as block '0000' at position 9) so to encode position information
        # we prefix each block with its location prior to hashing
        for i in range(len(blocks)):
            if '1' in blocks[i]:
                key = "%d%s" % (i, blocks[i])
                if key in self.table:
                    if self.limit is not None and len(self.table[key]) > self.limit:
                        self.table[key] = self.table[key][(self.limit // 2):]
                    self.table[key].append(obj_identifier)
                else:
                    self.table[key] = [obj_identifier]

    def get_best_match(self, new_obj):
        """
        Look up the stored identifier that best matches the given object
        """
        # first, split the object into blocks
        blocks = [new_obj[i:i+self.block_size] for i in range(0, len(new_obj), self.block_size)]
        # now hash each block and look up matches. Keep track of all candidates
        # that match at some hashes
        all_matches = []
        nnz = 0
        for i in range(len(blocks)):
            if '1' in blocks[i]:
                nnz += 1
                key = "%d%s" % (i, blocks[i])
                if key in self.table:
                    all_matches += self.table[key]
        # if the list of possible matches is empty, signal that no candidate
        # was found.
        if len(all_matches) == 0:
            return -1
        # if sampling is enabled, take self.random_samples samples from the
        # list of matches, and treat that as the list of candidates
        if self.random_samples is not None and len(all_matches) > self.random_samples:
            all_matches = np.random.choice(all_matches, size=self.random_samples, replace=False)
        # the candidate that appears the most times in the list of candidates
        # matches in the most indices and is thus the best choice
        match, count = Counter(all_matches).most_common(1)[0]
        if count / nnz > 0.5:
            return match
        else:
            # insufficient overlap could actually lead to increased memory use
            return -1

