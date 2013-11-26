import numpy as np
import random

from mrec.evaluation import metrics

from warp import WARPMFRecommender
from warp_fast import warp2_sample, apply_updates

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

    def __init__(self,d,gamma,C,batch_size=10,positive_thresh=0.00001,max_trials=50):
        WARPMFRecommender.__init__(self,d,gamma,C,batch_size,positive_thresh,max_trials)
        self.W = None

    def __str__(self):
        return 'WARP2MF(d={0},gamma={1},C={2})'.format(self.d,self.gamma,self.C)

    def _init(self,train,X):
        WARPMFRecommender._init(self,train)
        # W holds latent factors for each item feature
        num_features = X.shape[1]
        self.W = self.d**-0.5*np.random.random_sample((num_features,self.d))

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
        self._init(train,X)
        max_iters,validation_iters,validation = self.create_validation_set(train)

        precs = []
        tot_trials = 0
        for it in xrange(max_iters):
            if it % validation_iters == 0:
                # TODO: could have a stopping condition or budget on tot_trials
                print 'tot_trials',tot_trials
                tot_trials = 0
                prec = self.estimate_precision(train,validation,self.predict_ratings(X))
                precs.append(prec)
                print '{0}: validation precision = {1:.3f}'.format(it,precs[-1])
                if len(precs) > 3 and precs[-1] < precs[-2] and precs[-2] < precs[-3]:
                    print 'validation precision got worse twice, terminating'
                    break
            tot_trials += self.do_batch_update(train,X)

    def do_batch_update(self,train,X):
        u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,dW,trials = self.mini_batch(train,X)
        apply_updates(self.U,u_rows,dU,self.gamma,self.C)
        apply_updates(self.V,v_pos_rows,dV_pos,self.gamma,self.C)
        apply_updates(self.V,v_neg_rows,dV_neg,self.gamma,self.C)
        self.apply_matrix_update(self.W,dW)
        return trials

    def apply_matrix_update(self,W,dW):
            W += self.gamma*dW
            # ensure that ||W_k|| < C for all k
            p = np.sum(np.abs(W)**2,axis=-1)**0.5/self.C
            p[p<1] = 1
            W /= p[:,np.newaxis]

    def mini_batch(self,train,X):
        u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg = self._init_mini_batch()
        num_features = self.W.shape[0]
        dW = np.zeros((num_features,self.d))
        tot_trials = 0
        for ix in xrange(self.batch_size):
            u,i,j,N,trials = warp2_sample(self.U,self.V,self.W,X,train.data,train.indices,train.indptr,self.positive_thresh,self.max_trials)
            L = self.estimate_warp_loss(train,u,N)
            tot_trials += trials
            self._sgd_update(u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,dW,X,u,i,j,L,ix)

        return u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,dW,tot_trials

    def _sgd_update(self,u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,dW,X,u,i,j,L,ix):
        # compute gradient update
        # j is the violating item i.e. U[u]V[j]+U[u]XW[j] is too large
        # compared to U[u]V[i]+U[u]XW[i]
        u_rows[ix] = u
        dU[ix] = L*(self.V[i]-self.V[j] + (X[i]-X[j]).dot(self.W))
        v_pos_rows[ix] = i
        dV_pos[ix] = L*self.U[u]
        v_neg_rows[ix] = j
        dV_neg[ix] = -L*self.U[u]
        dW += L*np.atleast_2d(X[i]-X[j]).T.dot(np.atleast_2d(self.U[u]))

    def predict_ratings(self,X):
        def predict_ratings(users=None):
            if users is None:
                U = self.U
            else:
                U = np.asfortranarray(self.U[users,:])
            return U.dot(self.V.T + X.dot(self.W).T)
        return predict_ratings

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
    import cProfile
    cProfile.run('main()')
