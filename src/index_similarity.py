import numpy as np
from collections import Counter

class IndexSimilarityDetector():
    """
    Class that performs inexact matching of objects by looking up previously
    stored candidates by nonzero index. This provides a fast way of counting
    the number of nonzero indices in common between two vectors.

    In this implementation, objects are assumed to be represented as lists or
    arrays of (index, value) pairs.

    Optionally, matching candidates can be selected at random to improve
    performance.
    """

    def __init__(self, random_samples=None):
        self.table = {}
        self.random_samples = random_samples

    def add(self, new_obj, obj_identifier):
        """
        Add a new entry to the table of similarity candidates. The entry is
        identified using the provided identifier, which could be, for instance,
        an index into an array
        """
        for idx in new_obj:
            if idx not in self.table:
                self.table[idx] = [obj_identifier]
            else:
                self.table[idx].append(obj_identifier)

    def get_best_match(self, new_obj):
        """
        Look up the stored identifier that best matches the given object
        """
        all_matches = []
        # build up a list of all identifiers with matching nonzero indices.
        # May contain repeats.
        for idx in new_obj:
            if idx in self.table:
                all_matches += self.table[idx]
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
        if count / len(new_obj) > 0.5:
            return match
        else:
            # insufficient overlap could actually lead to increased memory use
            return -1

