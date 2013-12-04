"""
Recommender that gets candidates using an item similarity model
and then reranks them using a matrix factorization model.
"""

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np

from base_recommender import BaseRecommender

class RerankingRecommender(BaseRecommender):
    """
    A secondary recommender that combines an item similarity
    model and a matrix factorization one. The item similarity
    model is used to select candidate items for each user which
    are then reranked based on their latent factors.

    Parameters
    ==========
    item_similarity_recommender : mrec.item_similarity.recommender.ItemSimilarityRecommender
        The model used to select candidates.
    mf_recommender : mrec.mf.recommender.MatrixFactorizationRecommender
        The model used to rerank them.
    num_candidates : int (default: 100)
        The number of candidate items drawn from the first model for each user.
    """

    def __init__(self,item_similarity_recommender,mf_recommender,num_candidates=100):
        self.item_similarity_recommender = item_similarity_recommender
        self.mf_recommender = mf_recommender
        self.num_candidates = num_candidates
        self.description = 'RerankingRecommender({0},{1})'.format(self.item_similarity_recommender,self.mf_recommender)

    def _create_archive(self):
        archive = self.item_similarity_recommender._create_archive()
        archive['item_similarity_model'] = archive['model']
        archive.update(self.mf_recommender._create_archive())
        archive['mf_model'] = archive['model']
        tmp = self.item_similarity_recommender,self.mf_recommender
        self.item_similarity_model = self.mf_recommender = None
        m = pickle.dumps(self)
        self.item_similarity_model,self.mf_recommender = tmp
        archive['model'] = m
        return archive

    def _load_archive(self,archive):
        self.item_similarity_recommender = np.loads(str(archive['item_similarity_model']))
        self.item_similarity_recommender._load_archive(archive)
        self.mf_recommender = np.loads(str(archive['mf_model']))
        self.mf_recommender._load_archive(archive)

    def fit(self,train,item_features=None):
        """
        Fit both models to the training data.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix, shape = [num_users, num_items]
            The training user-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, required by some recommenders.

        Notes
        =====
        You are not obliged to call this, alternatively you can pass
        ready trained models to the RerankingRecommender constructor.
        """
        self.item_similarity_recommender.fit(train,item_features)
        self.mf_recommender.fit(train,item_features)

    def rerank(self,u,candidates,max_items,return_scores):
        """
        Use latent factors to rerank candidate recommended items for a user
        and return the highest scoring.

        Parameters
        ==========
        u : int
            Index of user for which to make recommendations.
        candidates : array like
            List of candidate item indices.
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
        r = self.mf_recommender.U[u].dot(self.mf_recommender.V[candidates].T)
        reranked = r.argsort()[:-1-max_items:-1]
        if return_scores:
            recs = [(candidates[i],r[i]) for i in reranked]
        else:
            recs = [candidates[i] for i in reranked]
        return recs

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
            Features for items in training set, required by some recommenders.

        Returns
        =======
        recs : list
            List of (idx,score) pairs if return_scores is True, else
            just a list of idxs.
        """
        candidates = self.item_similarity_recommender.recommend_items(dataset,u,self.num_candidates,return_scores=False)
        return self.rerank(u,candidates,max_items,return_scores=return_scores)

    def batch_recommend_items(self,
                              dataset,
                              max_items=10,
                              return_scores=True,
                              item_features=None):
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
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, required by some recommenders.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.
        """
        recs = self.item_similarity_recommender.batch_recommend_items(dataset,self.num_candidates,return_scores=False,item_features=item_features)
        for u,candidates in enumerate(recs):
            recs[u] = self.rerank(u,candidates,max_items,return_scores=return_scores)
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
        item_features : array_like, shape = [num_items, num_features]
            Features for items in training set, required by some recommenders.

        Returns
        =======
        recs : list of lists
            Each entry is a list of (idx,score) pairs if return_scores is True,
            else just a list of idxs.
        """
        recs = self.item_similarity_recommender.range_recommend_items(dataset,user_start,user_end,self.num_candidates,return_scores=False,item_features=item_features)
        for u,candidates in enumerate(recs):
            recs[u] = self.rerank(user_start+u,candidates,max_items,return_scores=return_scores)
        return recs

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix
    from mrec.item_similarity.knn import CosineKNNRecommender
    from mrec.mf.warp import WARPMFRecommender
    from mrec.reranking_recommender import RerankingRecommender

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    outfile = sys.argv[3]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    item_sim_model = CosineKNNRecommender(k=100)
    mf_model = WARPMFRecommender(d=80,gamma=0.01,C=100.0,max_iters=25000,validation_iters=1000,batch_size=10)
    recommender = RerankingRecommender(item_sim_model,mf_model,num_candidates=100)

    recommender.fit(train)

    save_recommender(recommender,outfile)

if __name__ == '__main__':
    main()

