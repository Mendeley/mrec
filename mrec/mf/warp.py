import numpy as np
import random

from recommender import MatrixFactorizationRecommender
from warp_fast import warp_sample, apply_updates, sample_positive_example, sample_violating_negative_example

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
    max_iters : int
        Terminate after this number of iterations even if validation loss is still decreasing.
    validation_iters : int
        Check validation loss once each validation_iters iterations, terminate if it
        has increased.
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Consider an item to be "positive" i.e. liked if its rating is at least this.
    max_trials : int
        Number of attempts allowed to find a violating negative example during updates.
        In practice it means that we optimize for ranks 1-max_trials.
    """

    def __init__(self,d,gamma,C,max_iters=10000,validation_iters=1000,batch_size=10,positive_thresh=0.00001,max_trials=50):
        self.d = d  # embedding dimension
        self.gamma = gamma # learning rate
        self.C = C  # regularization constant
        self.max_iters = max_iters
        self.validation_iters = validation_iters # check validation error after this many iterations
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials
        self.U = None
        self.V = None

    def __str__(self):
        return 'WARPMF(d={0},gamma={1},C={2},max_iters={3},validation_iters={4})'.format(self.d,self.gamma,self.C,self.max_iters,self.validation_iters)

    def _init(self,train):
        num_users,num_items = train.shape
        # precompute WARP loss for each possible rank
        self.warp_loss = np.ones(num_items)
        assert(num_items>1)
        for i in xrange(1,num_items):
            self.warp_loss[i] = self.warp_loss[i-1]+1.0/(i+1)  # L(i) = \sum_{0,i}{1/(i+1)}
        # initialize factors to small random values
        self.U = self.d**-0.5*np.random.random_sample((num_users,self.d))
        self.V = self.d**-0.5*np.random.random_sample((num_items,self.d))
        # ensure memory layout avoids extra allocation in dot product
        self.U = np.asfortranarray(self.U)
        # create validation samples for which we'll measure loss
        num_validation_samples = train.nnz/100
        # clip to a sensible range
        if num_validation_samples < 100:
            num_validation_samples = 100
        elif num_validation_samples > 1000:
            num_validation_samples = 1000
        self.validation_samples = [sample_positive_example(self.positive_thresh,num_users,train.data,train.indices,train.indptr) for _ in xrange(num_validation_samples)]

    def fit(self,train):
        """
        Learn factors from training set.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.
        sampler : mrec.mf.warp.Sampler (default: None)
            Sampler to provide rating pairs, if None a UniformUserSampler(train)
            will be used.
        """
        self._init(train)
        errs = []
        precs = []
        for it in xrange(self.max_iters):
            if it % self.validation_iters == 0:
                errs.append(self.validation_loss(train))
                print '{0}: validation loss = {1}'.format(it,errs[-1])
                if len(errs) > 3 and errs[-1] > errs[-2] and errs[-2] > errs[-3]:
                    print 'validation loss got worse twice, terminating'
                    break
            u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg = self.mini_batch(train)
            apply_updates(self.U,u_rows,dU,self.gamma,self.C)
            apply_updates(self.V,v_pos_rows,dV_pos,self.gamma,self.C)
            apply_updates(self.V,v_neg_rows,dV_neg,self.gamma,self.C)

    def mini_batch(self,train):
        u_rows = np.zeros(self.batch_size,dtype='int32')
        dU = np.zeros((self.batch_size,self.d),order='F')
        v_pos_rows = np.zeros(self.batch_size,dtype='int32')
        dV_pos = np.zeros((self.batch_size,self.d))
        v_neg_rows = np.zeros(self.batch_size,dtype='int32')
        dV_neg = np.zeros((self.batch_size,self.d))
        for ix in xrange(self.batch_size):
            u,i,j,N = warp_sample(self.U,self.V,train.data,train.indices,train.indptr,self.positive_thresh,self.max_trials)
            L = self.estimate_warp_loss(train,u,N)
            # compute gradient update
            # j is the violating item i.e. U[u].V[j] is too large compared to U[u].V[i]
            u_rows[ix] = u
            dU[ix] = L*(self.V[i]-self.V[j])
            v_pos_rows[ix] = i
            dV_pos[ix] = L*self.U[u]
            v_neg_rows[ix] = j
            dV_neg[ix] = -L*self.U[u]
        return u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg

    def estimate_warp_loss(self,train,u,N):
        num_items = train.shape[1]
        nnz = train.indptr[u+1]-train.indptr[u]
        estimated_rank = (num_items-nnz-1)/N
        return self.warp_loss[estimated_rank]

    def validation_loss(self,train):
        loss = 0
        for u,ix,i in self.validation_samples:
            begin = train.indptr[u]
            end = train.indptr[u+1]
            j,N = sample_violating_negative_example(self.U,self.V,train.data,train.indices,begin,end,u,ix,i,self.max_trials)
            if j >= 0:
                loss += self.estimate_warp_loss(train,u,N)
        return loss

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.sparse import fast_sparse_matrix

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    outfile = sys.argv[3]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    model = WARPMFRecommender(d=100,gamma=0.01,C=100.0,max_iters=10000,validation_iters=1000,batch_size=10)
    model.fit(train)

    save_recommender(model,outfile)

if __name__ == '__main__':
    main()
