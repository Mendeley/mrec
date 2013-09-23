class BaseRecommender(object):
    """
    Minimal interface to be implemented by recommenders.
    """

    def init(self,dataset):
        self.dataset = dataset
        self.num_users,self.num_items = self.dataset.shape

    def get_similar_items(self,j,max_similar_items=30):
        """
        Get the most similar items to a supplied item.

        Parameters
        ==========
        j : int
            Index of item for which to get similar items.
        max_similar_items : int
            Maximum number of similar items to return.

        Returns
        =======
        sims : list
            Sorted list of similar items, best first.  Each entry is
            a tuple of the form (i,score).
        """
        pass

    def recommend_items(self,u,max_items=10,return_scores=True):
        """
        Recommend new items for a user.

        Parameters
        ==========
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

    def batch_recommend_items(self,max_items=10,return_scores=True,show_progress=False):
        """
        Recommend new items for all users in the training dataset.

        Parameters
        ==========
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
            recs.append(self.recommend_items(u,max_items,return_scores))
        if show_progress:
            print
        return recs

    def range_recommend_items(self,user_start,user_end,max_items=10,return_scores=True):
        """
        Recommend new items for a range of users in the training dataset.
        Assumes you've already called train() to learn the similarity matrix.

        Parameters
        ==========
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
            recs.append(self.recommend_items(u,max_items,return_scores))
        return recs
