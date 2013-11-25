import numpy as np
import random

from mrec.evaluation import metrics

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
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Consider an item to be "positive" i.e. liked if its rating is at least this.
    max_trials : int
        Number of attempts allowed to find a violating negative example during updates.
        In practice it means that we optimize for ranks 1 to max_trials-1.
    """

    def __init__(self,d,gamma,C,batch_size=10,positive_thresh=0.00001,max_trials=50):
        self.d = d  # embedding dimension
        self.gamma = gamma # learning rate
        self.C = C  # regularization constant
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials
        self.U = None
        self.V = None

    def __str__(self):
        return 'WARPMF(d={0},gamma={1},C={2})'.format(self.d,self.gamma,self.C)

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

        validation_set = self.create_validation_set(train,num_validation_users)
        precs = []
        tot_trials = 0
        for it in xrange(max_iters):
            if it % validation_iters == 0:
                # TODO: could have a stopping condition or budget on tot_trials
                print 'tot_trials',tot_trials
                tot_trials = 0
                prec = self.estimate_precision(train,validation_set)
                precs.append(prec)
                print '{0}: validation precision = {1:.3f}'.format(it,precs[-1])
                if len(precs) > 3 and precs[-1] < precs[-2] and precs[-2] < precs[-3]:
                    print 'validation precision got worse twice, terminating'
                    break
            u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,trials = self.mini_batch(train)
            tot_trials += trials
            apply_updates(self.U,u_rows,dU,self.gamma,self.C)
            apply_updates(self.V,v_pos_rows,dV_pos,self.gamma,self.C)
            apply_updates(self.V,v_neg_rows,dV_neg,self.gamma,self.C)

    def create_validation_set(self,train,num_validation_users):
        """
        Hide and return half of the known items for validation users.
        """
        validation = dict()
        for u in xrange(num_validation_users):
            positive = np.where(train[u].data > 0)[0]
            hidden = random.sample(positive,positive.shape[0]/2)
            if hidden:
                train[u].data[hidden] = 0
                validation[u] = train[u].indices[hidden]
        return validation

    def mini_batch(self,train):
        u_rows = np.zeros(self.batch_size,dtype='int32')
        dU = np.zeros((self.batch_size,self.d),order='F')
        v_pos_rows = np.zeros(self.batch_size,dtype='int32')
        dV_pos = np.zeros((self.batch_size,self.d))
        v_neg_rows = np.zeros(self.batch_size,dtype='int32')
        dV_neg = np.zeros((self.batch_size,self.d))
        tot_trials = 0
        for ix in xrange(self.batch_size):
            u,i,j,N,trials = warp_sample(self.U,self.V,train.data,train.indices,train.indptr,self.positive_thresh,self.max_trials)
            L = self.estimate_warp_loss(train,u,N)
            # compute gradient update
            # j is the violating item i.e. U[u].V[j] is too large compared to U[u].V[i]
            u_rows[ix] = u
            dU[ix] = L*(self.V[i]-self.V[j])
            v_pos_rows[ix] = i
            dV_pos[ix] = L*self.U[u]
            v_neg_rows[ix] = j
            dV_neg[ix] = -L*self.U[u]
            tot_trials += trials
        return u_rows,dU,v_pos_rows,dV_pos,v_neg_rows,dV_neg,tot_trials

    def estimate_warp_loss(self,train,u,N):
        num_items = train.shape[1]
        nnz = train.indptr[u+1]-train.indptr[u]
        estimated_rank = (num_items-nnz-1)/N
        return self.warp_loss[estimated_rank]

    def estimate_precision(self,train,validation_set,k=30):
        """
        Compute prec@k for a sample of training users.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            The training data.
        k : int
            Measure precision@k.
        validation_set : dict or int
            Validation set over which we compute precision. Either supply
            a dict of user -> list of hidden items, or an integer n, in which
            case we simply evaluate against the training data for the first
            n users.

        Returns
        =======
        prec : float
            Precision@k computed over a sample of the training users.

        Notes
        =====
        At the moment this will underestimate the precision of real
        recommendations because we do not exclude training items with zero
        ratings from the top-k predictions evaluated.
        """
        if isinstance(validation_set,dict):
            have_validation_set = True
            users = validation_set.keys()
        elif isinstance(validation_set,(int,long)):
            have_validation_set = False
            users = range(validation_set)
        else:
            raise ValueError('validation_set must be dict or int')

        r = self.U[users,:].dot(self.V.T)
        prec = 0
        for ix,u in enumerate(users):
            ru = r[ix]
            predicted = ru.argsort()[::-1][:k]
            if have_validation_set:
                actual = validation_set[u]
            else:
                actual = train[u].indices[train[u].data > 0]
            prec += metrics.prec(predicted,actual,k)
        return prec/len(users)

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
