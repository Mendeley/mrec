import numpy as np
from scipy.sparse import csr_matrix

class BaseRecommender(object):
    """
    Minimal interface to be implemented by recommenders.
    """

    def recommend_items(self,dataset,u,max_items=10,return_scores=True):
        """
        Recommend new items for a user.

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
        pass

    def batch_recommend_items(self,dataset,max_items=10,return_scores=True,show_progress=False):
        """
        Recommend new items for all users in the training dataset.

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
        # default implementation, you may be able to optimize this for some recommenders.
        recs = []
        for u in xrange(self.num_users):
            if show_progress and u%1000 == 0:
               print u,'..',
            recs.append(self.recommend_items(dataset,u,max_items,return_scores))
        if show_progress:
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
        # default implementation, you may be able to optimize this for some recommenders.
        recs = []
        for u in xrange(user_start,user_end):
            recs.append(self.recommend_items(dataset,u,max_items,return_scores))
        return recs

    def _zero_known_item_scores(self,r,train):
        """
        Helper function to set predicted scores/ratings for training items
        to zero or less, to avoid recommending already known items.

        Parameters
        ==========
        r : numpy.ndarray or scipy.sparse.csr_matrix
            Predicted scores/ratings.
        train : scipy.sparse.csr_matrix
            The training user-item matrix, which can include zero-valued entries.

        Returns
        =======
        r_safe : scipy.sparse.csr_matrix
            r_safe is equal to r except that r[u,i] <= 0 for all u,i with entries
            in train.
        """
        col = train.indices
        if isinstance(r,csr_matrix):
            max_score = r.data.max()
        else:
            max_score = r.max()
        data = max_score * np.ones(col.shape)
        # build up the row (user) indices
        # - we can't just use row,col = train.nonzero() as this eliminates
        #   u,i for which train[u,i] has been explicitly set to zero
        row = np.zeros(col.shape)
        for u in xrange(train.shape[0]):
            start,end = train.indptr[u],train.indptr[u+1]
            if end > start:
                row[start:end] = u
        return r - csr_matrix((data,(row,col)),shape=r.shape)

