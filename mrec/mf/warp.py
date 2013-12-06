import numpy as np
import random

from mrec.evaluation import metrics

from recommender import MatrixFactorizationRecommender
from model.warp import WARP

class WARPMFRecommender(MatrixFactorizationRecommender):
    """
    Learn matrix factorization optimizing the WARP loss.

    Parameters
    ==========
    d : int
        Dimensionality of factors.
    gamma : float
        Learning rate.
    C : float
        Regularization constant.
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Consider an item to be "positive" i.e. liked if its rating is at least this.
    max_trials : int
        Number of attempts allowed to find a violating negative example during updates.
        In practice it means that we optimize for ranks 1 to max_trials-1.
    """

    def __init__(self,d,gamma,C,batch_size=10,positive_thresh=0.00001,max_trials=50):
        self.d = d
        self.gamma = gamma
        self.C = C
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials

    def fit(self,train,item_features=None):
        """
        Learn factors from training set.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.
        item_features : array_like, shape = [num_items, num_features]
            Features for each item in the dataset, ignored here.
        """
        max_iters,validation_iters,validation = self.create_validation_set(train)
        model = WARP(self.d,self.gamma,self.C,max_iters,validation_iters,self.batch_size,self.positive_thresh,self.max_trials)
        self.description = 'WARPMF({0})'.format(model)
        model.fit(train,validation)

        self.U = model.U_
        self.V = model.V_

    def create_validation_set(self,train):
        """
        Hide and return half of the known items for a sample of users,
        and estimate the number of sgd iterations to run.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.

        Returns
        =======
        max_iters : int
            Total number of sgd iterations to run.
        validation_iters : int
            Check progress after this many iterations.
        validation : dict
            Validation set.
        """
        # use 1% of users for validation, with a floor
        num_users = train.shape[0]
        num_validation_users = max(num_users/100,100)
        # ensure reasonable expected number of updates per validation user
        validation_iters = 100*num_users/num_validation_users
        # and reasonable number of validation cycles
        max_iters = 30*validation_iters

        print num_validation_users,'validation users'
        print validation_iters,'validation iters'
        print max_iters,'max_iters'

        validation = dict()
        for u in xrange(num_validation_users):
            positive = np.where(train[u].data > 0)[0]
            hidden = random.sample(positive,positive.shape[0]/2)
            if hidden:
                train[u].data[hidden] = 0
                validation[u] = train[u].indices[hidden]

        return max_iters,validation_iters,validation

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    outfile = sys.argv[3]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    model = WARPMFRecommender(d=100,gamma=0.01,C=100.0,batch_size=10)
    model.fit(train)

    save_recommender(model,outfile)

if __name__ == '__main__':
    main()
