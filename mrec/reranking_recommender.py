"""
Recommender that gets candidates using an item similarity model
and then reranks them using a matrix factorization model.
"""

from base_recommender import BaseRecommender

class RerankingRecommender(BaseRecommender):

    def __init__(self,item_similarity_recommender,mf_recommender,num_candidates=200):
        self.item_similarity_recommender = item_similarity_recommender
        self.mf_recommender = mf_recommender
        self.num_candidates = num_candidates

    def __str__(self):
        return 'RerankingRecommender({0},{1})'.format(self.item_similarity_recommender,self.mf_recommender)

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
        candidates = self.item_similarity_recommender.recommend_items(dataset,u,self.num_candidates,return_scores=False)
        return self.rerank(u,candidates,max_items,return_scores=return_scores)

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
        recs = self.item_similarity_recommender.batch_recommend_items(dataset,self.num_candidates,return_scores=False)
        for u,candidates in enumerate(recs):
            recs[u] = self.rerank(u,candidates,max_items,return_scores=return_scores)
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
        recs = self.item_similarity_recommender.range_recommend_items(dataset,user_start,user_end,self.num_candidates,return_scores=False)
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

    item_sim_model = CosineKNNRecommender(k=50)
    item_sim_model.fit(train)

    mf_model = WARPMFRecommender(d=100,gamma=0.01,C=100.0,max_iters=15000,validation_iters=1000,batch_size=10)
    mf_model.fit(train)

    recommender = RerankingRecommender(item_sim_model,mf_model)

    recs = recommender.batch_recommend_items(train,max_items=20)
    recs2 = recommender.range_recommend_items(train,2,20,max_items=20)

    for u in xrange(2,20):
        knn = item_sim_model.recommend_items(train,u,max_items=20)
        print 'cosine knn:'
        for i,score in knn:
            print '{0}\t{1}\t{2}'.format(u,i,score)
        reranked = recommender.recommend_items(train,u,max_items=20)
        print 'recs[u]'
        print recs[u]
        print 'reranked'
        print reranked
        print 'recs2[u-2]'
        print recs2[u-2]
        assert(reranked == recs[u])
        assert(reranked == recs2[u-2])
        print 'reranked:'
        for i,score in reranked:
            print '{0}\t{1}\t{2}'.format(u,i,score)

if __name__ == '__main__':
    main()

