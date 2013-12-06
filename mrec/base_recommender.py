try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
from scipy.sparse import csr_matrix

class BaseRecommender(object):
    """
    Minimal interface to be implemented by recommenders, along with
    some helper methods. A concrete recommender must implement the
    recommend_items() method and should provide its own implementation
    of __str__() so that it can be identified when printing results.

    Notes
    =====
    In most cases you should inherit from either
    `mrec.mf.recommender.MatrixFactorizationRecommender` or
    `mrec.item_similarity.recommender.ItemSimilarityRecommender`
    and *not* directly from this class.

    These provide more efficient implementations of save(), load()
    and the batch methods to recommend items.
    """

    def recommend_items(self,dataset,u,max_items=10,return_scores=True,item_features=None):
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
        item_features : array_like, shape = [num_items, num_features]
            Optionally supply features for each item in the dataset.

        Returns
        =======
        recs : list
            List of (idx,score) pairs if return_scores is True, else
            just a list of idxs.
        """
        raise NotImplementedError('you must implement recommend_items()')

    def fit(self,train,item_features=None):
        """
        Train on supplied data. In general you will want to
        implement this rather than computing recommendations on
        the fly.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix or mrec.sparse.fast_sparse_matrix, shape = [num_users, num_items]
            User-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, required by some recommenders.
        """
        raise NotImplementedError('you should implement fit()')

    def save(self,filepath):
        """
        Serialize model to file.

        Parameters
        ==========
        filepath : str
            Filepath to write to, which must have the '.npz' suffix.

        Notes
        =====
        Internally numpy.savez may be used to serialize the model and
        this would add the '.npz' suffix to the supplied filepath if
        it were not already present, which would most likely cause errors
        in client code.
        """
        if not filepath.endswith('.npz'):
            raise ValueError('invalid filepath {0}, must have ".npz" suffix'.format(filepath))

        archive = self._create_archive()
        if archive:
            np.savez(filepath,**archive)
        else:
            pickle.dump(self,open(filepath,'w'))

    def _create_archive(self):
        """
        Optionally return a dict of fields to be serialized
        in a numpy archive: this lets you store arrays efficiently
        by separating them from the model itself.

        Returns
        =======
        archive : dict
            Fields to serialize, must include the model itself
            under the key 'model'.
        """
        pass

    @staticmethod
    def load(filepath):
        """
        Load a recommender model from file after it has been serialized with
        save().

        Parameters
        ==========
        filepath : str
            The filepath to read from.
        """
        r = np.load(filepath)
        if isinstance(r,BaseRecommender):
            model = r
        else:
            model = np.loads(str(r['model']))
            model._load_archive(r)  # restore any fields serialized separately
        return model

    def _load_archive(archive):
        """
        Load fields from a numpy archive.

        Notes
        =====
        This is called by the static load() method and should be used
        to restore the fields returned by _create_archive().
        """
        pass

    @staticmethod
    def read_recommender_description(filepath):
        """
        Read a recommender model description from file after it has
        been saved by save(), without loading any additional
        associated data into memory.

        Parameters
        ==========
        filepath : str
            The filepath to read from.
        """
        r = np.load(filepath,mmap_mode='r')
        if isinstance(r,BaseRecommender):
            model = r
        else:
            model = np.loads(str(r['model']))
        return str(model)

    def __str__(self):
        if hasattr(self,'description'):
            return self.description
        return 'unspecified recommender: you should set self.description or implement __str__()'

    def batch_recommend_items(self,
                              dataset,
                              max_items=10,
                              return_scores=True,
                              show_progress=False,
                              item_features=None):
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
        item_features : array_like, shape = [num_items, num_features]
            Optionally supply features for each item in the dataset.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.

        Notes
        =====
        This provides a default implementation, you will be able to optimize
        this for most recommenders.
        """
        recs = []
        for u in xrange(self.num_users):
            if show_progress and u%1000 == 0:
               print u,'..',
            recs.append(self.recommend_items(dataset,u,max_items,return_scores))
        if show_progress:
            print
        return recs

    def range_recommend_items(self,
                              dataset,
                              user_start,
                              user_end,
                              max_items=10,
                              return_scores=True,
                              item_features=None):
        """
        Recommend new items for a range of users in the training dataset.

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
        item_features : array_like, shape = [num_items, num_features]
            Optionally supply features for each item in the dataset.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.

        Notes
        =====
        This provides a default implementation, you will be able to optimize
        this for most recommenders.
        """
        return [self.recommend_items(dataset,u,max_items,return_scores) for u in xrange(user_start,user_end)]

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

