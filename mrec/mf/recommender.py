"""
Base class for recommenders that work
by matrix factorization.
"""

import numpy as np
from itertools import izip
from scipy.sparse import csr_matrix

from mrec.base_recommender import BaseRecommender

class MatrixFactorizationRecommender(BaseRecommender):
    """
    Base class for matrix factorization recommenders.
    """

    def __str__(self):
        if hasattr(self,'description'):
            return self.description
        return 'MatrixFactorizationRecommender'

    def fit(self,train):
        """
        Learn user and item factors from training dataset.

        Parameters
        ==========
        train : scipy sparse matrix
          The user-item matrix.
        """
        pass

    def load_factors(self,user_factor_filepath,item_factor_filepath,fmt):
        """
        Load precomputed user and item factors from file.

        Parameters
        ==========
        user_factor_filepath : str
            Filepath to tsv file holding externally computed user factors. Can be
            TSV, Matrix Market or numpy array serialized with numpy.save().
        item_factor_filepath : str
            Filepath to TSV file holding externally computed item factors. Can be
            TSV, Matrix Market or numpy array serialized with numpy.save().
        fmt : str: npy, mm or tsv
            File format: numpy array, Matrix Market or TSV.  Each line of TSV input
            should contain all of the factors for a single user or item.
        """
        if fmt == 'npy':
            self.U = np.load(user_factor_filepath)
            self.V = np.load(item_factor_filepath)
        elif fmt == 'mm':
            self.U = mmread(user_factor_filepath)
            self.V = mmread(item_factor_filepath)
        elif fmt == 'tsv':
            self.U = np.loadtxt(user_factor_filepath)
            self.V = np.loadtxt(item_factor_filepath)
        else:
            raise ValueError('unknown input format {0}'.format(fmt))
        # ensure that memory layout avoids extra allocation in dot product
        self.U = np.asfortranarray(self.U)

    def recommend_items(self,dataset,u,max_items=10,return_scores=True):
        """
        Recommend up to max_items most highly recommended items for user u.
        Assumes you've already called fit() to learn the factors.

        Parameters
        ==========
        dataset : scipy.sparse.csr_matrix
            User-item matrix containing known items.
        u : int
            Index of user for which to make recommendations.
        max_items : int
            Maximum number of recommended items to return.
        return_scores : bool
            If true return a score along with each recommended item.

        Returns
        =======
        recs : list
            List of (idx,score) pairs if return_scores is True, else
            just a list of idxs.
        """
        r = self.U[u].dot(self.V.T)
        known_items = set(dataset[u].indices)
        recs = []
        for i in r.argsort()[::-1]:
            if i not in known_items:
                if return_scores:
                    recs.append((i,r[i]))
                else:
                    recs.append(i)
                if len(recs) >= max_items:
                    break
        return recs

    def batch_recommend_items(self,dataset,max_items=10,return_scores=True):
        """
        Recommend new items for all users in the training dataset.  Assumes
        you've already called fit() to learn the similarity matrix.

        Parameters
        ==========
        dataset : scipy.sparse.csr_matrix
            User-item matrix containing known items.
        max_items : int
            Maximum number of recommended items to return.
        return_scores : bool
            If true return a score along with each recommended item.
        show_progress: bool
            If true print something to stdout to show progress.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.
        """
        r = self.U.dot(self.V.T)
        # make the predicted score for all known items
        # zero or less by substracting the max score from them
        max_score = r.max()  # highest predicted score
        row,col = dataset.nonzero()  # locations of known items
        data = max_score * np.ones(row.shape)
        r = np.array(r - csr_matrix((data,(row,col)),shape=r.shape))
        recs = [[] for u in xrange(self.num_users)]
        for u in xrange(self.num_users):
            ru = r[u]
            if u%1000 == 0:
               print u,'..',
            if return_scores:
                recs[u] = [(i,ru[i]) for i in ru.argsort()[::-1] if ru[i] > 0][:max_items]
            else:
                recs[u] = [i for i in ru.argsort()[::-1] if ru[i] > 0][:max_items]
        print
        return recs

    def range_recommend_items(self,dataset,user_start,user_end,max_items=10,return_scores=True):
        """
        Recommend new items for a range of users in the training dataset.
        Assumes you've already called fit() to learn the similarity matrix.

        Parameters
        ==========
        dataset : scipy.sparse.csr_matrix
            User-item matrix containing known items.
        user_start : int
            Index of first user in the range to recommend.
        user_end : int
            Index one beyond last user in the range to recommend.
        max_items : int
            Maximum number of recommended items to return.
        return_scores : bool
            If true return a score along with each recommended item.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.
        """
        data_subset = dataset[user_start:user_end,:]
        r = self.U[user_start:user_end,:].dot(self.V.T)
        # make the predicted score for all known items
        # zero or less by substracting the max score from them
        max_score = r.max()  # highest predicted score
        row,col = data_subset.nonzero()  # locations of known items
        data = max_score * np.ones(row.shape)
        r = np.array(r - csr_matrix((data,(row,col)),shape=r.shape))
        recs = [[] for u in xrange(user_start,user_end)]
        for u in xrange(user_start,user_end):
            ux = u - user_start
            ru = r[ux]
            if return_scores:
                recs[ux] = [(i,ru[i]) for i in ru.argsort()[::-1] if ru[i] > 0][:max_items]
            else:
                recs[ux] = [i for i in ru.argsort()[::-1] if ru[i] > 0][:max_items]
        return recs
