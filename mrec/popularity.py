"""
Trivial unpersonalized item popularity recommender
intended to provide a baseline for evaluations.
"""

import numpy as np

from base_recommender import BaseRecommender
from sparse import fast_sparse_matrix

class ItemPopularityRecommender(BaseRecommender):
    """
    Create an unpersonalized item popularity recommender, useful
    to provide a baseline for comparison with a "real" one.

    Parameters
    ----------

    method : 'count', 'sum', 'avg' or 'thresh' (default: 'count')
        How to calculate the popularity of an item based on its ratings
        from all users:
        count - popularity is its total number of ratings of any value
        sum - popularity is the sum of its ratings
        avg - popularity is its mean rating
        thresh - popularity is its number of ratings higher than thresh
    thresh : float, optional
        The threshold used by the 'thresh' method of calculating item
        popularity.
    """

    def __init__(self,method='count',thresh=0):
        self.description = 'ItemPop'
        if method not in ['count','sum','avg','thresh']:
            raise ValueError('invalid value for method parameter')
        self.method = method
        self.thresh = thresh

    def fit(self,dataset,item_features=None):
        """
        Compute the most popular items using the method specified
        in the constructor.

        Parameters
        ----------
        dataset : scipy sparse matrix or mrec.sparse.fast_sparse_matrix
            The user-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, ignored here.
        """
        if isinstance(dataset,fast_sparse_matrix):
            d = dataset.X.tocsc()
        else:
            d = dataset.tocsc()
        if self.method == 'count':
            # count the total number of ratings for each item
            popularity = [(d[:,i].nnz,i) for i in xrange(d.shape[1])]
        elif self.method == 'sum':
            # find the sum of the ratings for each item
            popularity = [(d[:,i].sum(),i) for i in xrange(d.shape[1])]
        elif self.method == 'avg':
            # find the mean rating for each item
            popularity = [(d[:,i].mean(),i) for i in xrange(d.shape[1])]
        elif self.method == 'thresh':
            # count the number of ratings above thresh for each item
            popularity = [(sum(d[:,i].data>self.thresh),i) for i in xrange(d.shape[1])]
        popularity.sort(reverse=True)
        self.pop_items = [(i,c) for (c,i) in popularity]

    def recommend_items(self,dataset,u,max_items=10,return_scores=True,item_features=None):
        """
        Recommend new items for a user.  Assumes you've already called
        fit().

        Parameters
        ----------
        dataset : scipy.sparse.csr_matrix
            User-item matrix containing known items.
        u : int
            Index of user for which to make recommendations (for
            compatibility with other recommenders).
        max_items : int
            Maximum number of recommended items to return.
        return_scores : bool
            If true return a score along with each recommended item.
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, ignored here.

        Returns
        -------
        recs : list
            List of (idx,score) pairs if return_scores is True, else
            just a list of idxs.
        """
        known_items = set(dataset[u].indices)
        recs = []
        for i,c in self.pop_items:
            if i not in known_items:
                if return_scores:
                    recs.append((i,c))
                else:
                    recs.append(i)
                if len(recs) >= max_items:
                    break
        return recs

    def __str__(self):
        return self.description
