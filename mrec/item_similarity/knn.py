"""
Brute-force k-nearest neighbour recommenders
intended to provide evaluation baselines.
"""

import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils.extmath import safe_sparse_dot

from recommender import ItemSimilarityRecommender

class KNNRecommender(ItemSimilarityRecommender):
    """
    Abstract base class for k-nn recommenders.  You must supply an
    implementation of the compute_all_similarities() method.

    Parameters
    ==========
    k : int
        The number of nearest neighbouring items to retain
    """

    def __init__(self,k):
        self.k = k

    def compute_similarities(self,dataset,j):
        A = dataset.X
        a = dataset.fast_get_col(j)
        d = self.compute_all_similarities(A,a)
        d[j] = 0  # zero out self-similarity
        # now zero out similarities for all but top-k items
        nn = d.argsort()[-1:-1-self.k:-1]
        w = np.zeros(A.shape[1])
        w[nn] = d[nn]
        return w

    def compute_all_similarities(self,A,a):
        """
        Compute similarity scores between item vector a
        and all the rows of A.

        Parameters
        ==========
        A : scipy.sparse.csr_matrix
            Matrix of item vectors.
        a : array_like
            The item vector to be compared to each row of A.

        Returns
        =======
        similarities : numpy.ndarray
            Vector of similarity scores.
        """
        pass

class DotProductKNNRecommender(KNNRecommender):
    """
    Similarity between two items is their dot product
    (i.e. cooccurrence count if input data is binary).
    """

    def compute_all_similarities(self,A,a):
        return A.T.dot(a).toarray().flatten()

    def __str__(self):
        return 'DotProductKNNRecommender(k={0})'.format(self.k)

class CosineKNNRecommender(KNNRecommender):
    """
    Similarity between two items is their cosine distance.
    """

    def compute_all_similarities(self,A,a):
        return cosine_similarity(A.T,a.T).flatten()

    def __str__(self):
        return 'CosineKNNRecommender(k={0})'.format(self.k)

class AdjustedCosineKNNRecommender(KNNRecommender):
    """
    Similarity between two items is the adjusted cosine similarity:
    this is the cosine similarity between mean-centered vectors over the
    subset of users who have rated both items.

    See Sarwar et al, 'Item-Based Collaborative Filtering Recommendation Algorithms' (WWW '10).
    """


    def compute_all_similarities(self, A, a):

        # have we been training using this recommender before?
        _cached_deltas = getattr(self, '_cached_deltas', None)

        if _cached_deltas is None:
            # First, we need to normalize ratings in A --
            # divide through the set ratings by the mean rating of rated items.
            #
            # First, we need the mean - the sum over the count.
            #
            mean = A.sum(axis=1) / A.astype(bool).sum(axis=1).astype(float)
            #
            # we now want a sparse matrix where zero elements in the original matrix
            # are still zero, and non-zero elements are equal to the user mean.
            # we can get this by multiplying a diagonal matrix (as mean) against a
            # binarized version of the original matrix
            #
            # Why the diagonal matrix? Because this is a sparse matrix, normal broadcasting rules
            # don't apply, so we have to do the matrix multiplication explicitly.
            #
            dim = max(mean.shape)
            diag = scipy.sparse.lil_matrix((dim, dim))
            diag.setdiag(np.asarray(mean.flatten())[0])
            delta_A = diag * A.astype(bool).astype(float)
            self._cached_deltas = delta_A

        # now, finally, we can get the mean-centered values:
        norm_A = A - self._cached_deltas
        return cosine_similarity(norm_A.T, a.T).flatten()

    def __str__(self):
        return 'AdjustedCosineKNNRecommender(k={0})'.format(self.k)

class JaccardKNNRecommender(KNNRecommender):
    """
    Similarity between two items is the Jaccard similarity (intersection/union) between
    vectors of booleans, binarized as (zero, greater_than_zero).

    This is fiercely slow. In production, you likely don't want to be using this.
    """
    def compute_all_similarities(self,A,a):
        _cached_bool = getattr(self, '_cached_bool', None)
        _counts = getattr(self, '_counts', None)

        if _cached_bool == None:
            _cached_bool = A.astype(bool).T
            self._cached_bool = _cached_bool

        sims = []
        ba = a.astype(bool).T
        bsum = ba.sum()

        for idx, row in enumerate(self._cached_bool):
            union = (row + ba).sum()
            count = bsum + row.sum()
            intersection = count-union
            sims.append(float(intersection) / union)
        return np.asarray(sims)

    def __str__(self):
        return 'JaccardKNNRecommender(k={0})'.format(self.k)

if __name__ == '__main__':

    # use knn models like this:

    import random
    import StringIO
    from mrec import load_fast_sparse_matrix

    random.seed(0)

    print 'loading test data...'
    data = """\
%%MatrixMarket matrix coordinate real general
3 5 9
1	1	1
1	2	1
1	3	1
1	4	1
2	2	1
2	3	1
2	5	1
3	3	1
3	4	1
"""
    print data
    dataset = load_fast_sparse_matrix('mm',StringIO.StringIO(data))
    num_users,num_items = dataset.shape

    model = CosineKNNRecommender(k=2)

    num_samples = 2

    def output(i,j,val):
        # convert back to 1-indexed
        print '{0}\t{1}\t{2:.3f}'.format(i+1,j+1,val)

    print 'computing some item similarities...'
    print 'item\tsim\tweight'
    # if we want we can compute these individually without calling fit()
    for i in random.sample(xrange(num_items),num_samples):
        for j,weight in model.get_similar_items(i,max_similar_items=2,dataset=dataset):
            output(i,j,weight)

    print 'learning entire similarity matrix...'
    # more usually we just call train() on the entire dataset
    model = CosineKNNRecommender(k=2)
    model.fit(dataset)
    print 'making some recommendations...'
    print 'user\trec\tscore'
    for u in random.sample(xrange(num_users),num_samples):
        for i,score in model.recommend_items(dataset.X,u,max_items=10):
            output(u,i,score)

    print 'making batch recommendations...'
    recs = model.batch_recommend_items(dataset.X)
    for u in xrange(num_users):
        for i,score in recs[u]:
            output(u,i,score)

    print 'making range recommendations...'
    for start,end in [(0,2),(2,3)]:
        recs = model.range_recommend_items(dataset.X,start,end)
        for u in xrange(start,end):
            for i,score in recs[u-start]:
                output(u,i,score)
