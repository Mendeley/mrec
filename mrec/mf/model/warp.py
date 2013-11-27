import numpy as np
import random
from itertools import izip

from mrec.evaluation import metrics

from warp_fast import warp_sample, apply_updates

class WARPBatchUpdate(object):
    """Collection of arrays to hold a batch of sgd updates"""

    def __init__(self,batch_size,d):
        self.u = np.zeros(batch_size,dtype='int32')
        self.dU = np.zeros((batch_size,d),order='F')
        self.v_pos = np.zeros(batch_size,dtype='int32')
        self.dV_pos = np.zeros((batch_size,d))
        self.v_neg = np.zeros(batch_size,dtype='int32')
        self.dV_neg = np.zeros((batch_size,d))

    def set_update(self,ix,u,v_pos,v_neg,dU,dV_pos,dV_neg):
        self.u[ix] = u
        self.dU[ix] = dU
        self.v_pos[ix] = v_pos
        self.dV_pos[ix] = dV_pos
        self.v_neg[ix] = v_neg
        self.dV_neg[ix] = dV_neg

class WARPDecomposition(object):

    def __init__(self,num_rows,num_cols,d):
        # initialize factors to small random values
        self.U = d**-0.5*np.random.random_sample((num_rows,d))
        self.V = d**-0.5*np.random.random_sample((num_cols,d))
        # ensure memory layout avoids extra allocation in dot product
        self.U = np.asfortranarray(self.U)

    def compute_update(self,updates,ix,u,i,j,L):
        # compute gradient update
        # j is the violating col i.e. U[u].V[j] is too large compared to U[u].V[i]
        dU = L*(self.V[i]-self.V[j])
        dV_pos = L*self.U[u]
        dV_neg = -L*self.U[u]
        updates.set_update(ix,u,i,j,dU,dV_pos,dV_neg)

    def apply_updates(self,updates,gamma,C):
        apply_updates(self.U,updates.u,updates.dU,gamma,C)
        apply_updates(self.V,updates.v_pos,updates.dV_pos,gamma,C)
        apply_updates(self.V,updates.v_neg,updates.dV_neg,gamma,C)

    def reconstruct(self,rows):
        if rows is None:
            U = self.U
        else:
            U = np.asfortranarray(self.U[rows,:])
        return U.dot(self.V.T)

class WARP(object):
    """
    Learn low-dimensional embedding optimizing the WARP loss.

    Parameters
    ==========
    d : int
        Embedding dimension.
    gamma : float
        Learning rate.
    C : float
        Regularization constant.
    max_iters : int
        Maximum number of SGD updates.
    validation_iters : int
        Number of SGD updates between checks for stopping condition.
    batch_size : int
        Mini batch size for SGD updates.
    positive_thresh: float
        Training entries below this are treated as zero.
    max_trials : int
        Number of attempts allowed to find a violating negative example during
        training updates. This means that in practice we optimize for ranks 1
        to max_trials-1.

    Attributes
    ==========
    U_ : numpy.ndarray
        Row factors.
    V_ : numpy.ndarray
        Column factors.
    """

    def __init__(self,
                 d,
                 gamma,
                 C,
                 max_iters,
                 validation_iters,
                 batch_size=10,
                 positive_thresh=0.00001,
                 max_trials=50):
        self.d = d
        self.gamma = gamma
        self.C = C
        self.max_iters = max_iters
        self.validation_iters = validation_iters
        self.batch_size = batch_size
        self.positive_thresh = positive_thresh
        self.max_trials = max_trials

    def __str__(self):
        return 'WARP(d={0},gamma={1},C={2})'.format(self.d,self.gamma,self.C)

    def fit(self,train,validation=None):
        """
        Learn factors from training set. The dot product of the factors
        reconstructs the training matrix approximately, minimizing the
        WARP ranking loss relative to the original data.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            Training matrix to be factorized.
        validation : dict or int
            Validation set to control early stopping, based on precision@30.
            The dict should have the form row->[cols] where the values in cols
            are those we expected to be highly ranked in the reconstruction of
            row. If an int is supplied then instead we evaluate precision
            against the training data for the first validation rows.

        Returns
        =======
        self : object
            This model itself.
        """
        num_rows,num_cols = train.shape
        decomposition = WARPDecomposition(num_rows,num_cols,self.d)
        updates = WARPBatchUpdate(self.batch_size,self.d)
        self.precompute_warp_loss(num_cols)

        self._fit(decomposition,updates,train,validation)

        self.U_ = decomposition.U
        self.V_ = decomposition.V

        return self

    def _fit(self,decomposition,updates,train,validation):
        precs = []
        tot_trials = 0
        for it in xrange(self.max_iters):
            if it % self.validation_iters == 0:
                print 'tot_trials',tot_trials
                tot_trials = 0
                prec = self.estimate_precision(decomposition,train,validation)
                precs.append(prec)
                print '{0}: validation precision = {1:.3f}'.format(it,precs[-1])
                if len(precs) > 3 and precs[-1] < precs[-2] and precs[-2] < precs[-3]:
                    print 'validation precision got worse twice, terminating'
                    break
            tot_trials += self.compute_updates(train,decomposition,updates)
            decomposition.apply_updates(updates,self.gamma,self.C)

    def precompute_warp_loss(self,num_cols):
        """
        Precompute WARP loss for each possible rank:

            L(i) = \sum_{0,i}{1/(i+1)}
        """
        assert(num_cols>1)
        self.warp_loss = np.ones(num_cols)
        for i in xrange(1,num_cols):
            self.warp_loss[i] = self.warp_loss[i-1]+1.0/(i+1)

    def compute_updates(self,train,decomposition,updates):
        tot_trials = 0
        for ix in xrange(self.batch_size):
            u,i,j,N,trials = warp_sample(decomposition.U,
                                         decomposition.V,
                                         train.data,
                                         train.indices,
                                         train.indptr,
                                         self.positive_thresh,
                                         self.max_trials)
            tot_trials += trials
            L = self.estimate_warp_loss(train,u,N)
            decomposition.compute_update(updates,ix,u,i,j,L)
        return tot_trials

    def estimate_warp_loss(self,train,u,N):
        num_cols = train.shape[1]
        nnz = train.indptr[u+1]-train.indptr[u]
        estimated_rank = (num_cols-nnz-1)/N
        return self.warp_loss[estimated_rank]

    def estimate_precision(self,decomposition,train,validation,k=30):
        """
        Compute prec@k for a sample of training rows.

        Parameters
        ==========
        decomposition : WARPDecomposition
            The current decomposition.
        train : scipy.sparse.csr_matrix
            The training data.
        k : int
            Measure precision@k.
        validation : dict or int
            Validation set over which we compute precision. Either supply
            a dict of row -> list of hidden cols, or an integer n, in which
            case we simply evaluate against the training data for the first
            n rows.

        Returns
        =======
        prec : float
            Precision@k computed over a sample of the training rows.

        Notes
        =====
        At the moment this will underestimate the precision of real
        recommendations because we do not exclude training cols with zero
        ratings from the top-k predictions evaluated.
        """
        if isinstance(validation,dict):
            have_validation_set = True
            rows = validation.keys()
        elif isinstance(validation,(int,long)):
            have_validation_set = False
            rows = range(validation)
        else:
            raise ValueError('validation must be dict or int')

        r = decomposition.reconstruct(rows)
        prec = 0
        for u,ru in izip(rows,r):
            predicted = ru.argsort()[::-1][:k]
            if have_validation_set:
                actual = validation[u]
            else:
                actual = train[u].indices[train[u].data > 0]
            prec += metrics.prec(predicted,actual,k)
        return float(prec)/len(rows)

