import numpy as np

from warp import WARPMFRecommender
from model.warp2 import WARP2

class WARP2MFRecommender(WARPMFRecommender):
    """
    Learn matrix factorization optimizing the WARP loss
    with item features as well as user-item training data.

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

    def __str__(self):
        return 'WARP2MF(d={0},gamma={1},C={2})'.format(self.d,self.gamma,self.C)

    def fit(self,train,X):
        """
        Learn factors from training set and item features.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.
        X : numpy.ndarray.
            Item features.
        """
        max_iters,validation_iters,validation = self.create_validation_set(train)
        model = WARP2(self.d,self.gamma,self.C,max_iters,validation_iters,self.batch_size,self.positive_thresh,self.max_trials)
        self.description = 'WARP2MF({0})'.format(model)
        model.fit(train,X,validation)

        self.U = model.U_
        self.V = model.V_
        self.W = model.W_

    def predict_ratings(self,X,users=None):
        if users is None:
            U = self.U
        else:
            U = np.asfortranarray(self.U[users,:])
        return U.dot(self.V.T + X.dot(self.W).T)

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    feature_file = sys.argv[3]
    outfile = sys.argv[4]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)
    # load item features as numpy array
    X = load_sparse_matrix('tsv',feature_file).toarray()
    num_items = train.shape[1]
    X = X[:num_items,:]

    model = WARP2MFRecommender(d=100,gamma=0.01,C=100.0,batch_size=10)
    model.fit(train,X)

    save_recommender(model,outfile)

if __name__ == '__main__':
    main()
