import random

class TSVParser(object):
    """
    Parses tsv input: user, item, score.

    Parameters
    ----------
    thresh : float (default: 0)
        Set scores below this to zero.
    binarize : bool (default: False)
        If True, set all non-zero scores to 1.
    """

    def __init__(self,thresh=0,binarize=False,delimiter='\t'):
        self.thresh = thresh
        self.binarize = binarize
        self.delimiter = delimiter

    def parse(self,line):
        parts = line.strip().split(self.delimiter)
        user,item,count = parts[:3]
        val = float(count)
        if val >= self.thresh:
            if self.binarize:
                val = 1
        else:
            val = 0
        return int(user),(int(item),val)

class SplitCreator(object):
    """
    Split ratings for a user randomly into train
    and test groups.  Only items with positive scores
    will be included in the test group.

    Parameters
    ----------
    test_size : float
        If test_size >= 1 this specifies the absolute number
        of items to put in the test group; if test_size < 1
        then this specifies the test proportion.
    normalize : bool (default: False)
        If True, scale training scores for each user to have unit norm.
    discard_zeros : bool (default: False)
        If True then discard items with zero scores, if
        False then retain them in the training group.This
        should normally be False as such items have been seen
        (if not liked) and so the training set should include
        them so that it can be used to determine which items
        are actually novel at recommendation time.
    sample_before_thresholding : bool (default: False)
        If True then consider any item seen by the user for
        inclusion in the test group, even though only items
        with positive scrore will be selected. If the input
        includes items with zero scores this means that the
        test set may be smaller than the requested size for
        some users, even though they have apparently seen
        enough items.
    """

    def __init__(self,test_size,normalize=False,discard_zeros=False,sample_before_thresholding=False):
        self.test_size = test_size
        self.normalize = normalize
        self.discard_zeros = discard_zeros
        self.sample_before_thresholding = sample_before_thresholding

    def handle(self,vals):
        if self.sample_before_thresholding:
            train,test = self.split(vals)
        else:
            train,test = self.stratified_split(vals)
        train = [(v,c) for v,c in train if not self.discard_zeros or c > 0]
        test = [(v,c) for v,c in test if c > 0]
        if self.normalize:
            norm = sum(c*c for v,c in train)**0.5
            if norm > 0:
                train = [(v,c/norm) for v,c in train]
        return train,test

    def pos_neg_vals(self,vals):
        vals = list(vals)
        pos = [(v,c) for v,c in vals if c > 0]
        neg = [(v,0) for v,c in vals if c == 0]
        return pos,neg

    def split(self,vals):
        random.shuffle(vals)
        num_train = self.num_train(vals)
        return vals[:num_train],vals[num_train:]

    def stratified_split(self,vals):
        pos,neg = self.pos_neg_vals(vals)
        random.shuffle(pos)
        train = pos[:self.num_train(pos)]
        if not self.discard_zeros:
            random.shuffle(neg)
            train.extend(neg[:self.num_train(neg)])
            random.shuffle(train)
        test = pos[self.num_train(pos):]
        return train,test

    def num_train(self,vals):
        if self.test_size >= 1:
            return len(vals)-self.test_size
        return int(len(vals)*(1.0-self.test_size))
