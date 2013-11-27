import numpy as np
import random

from warp import WARPBatchUpdate, WARPDecomposition, WARP
from warp_fast import warp2_sample, apply_updates

class WARP2BatchUpdate(WARPBatchUpdate):
    """Collection of arrays to hold a batch of sgd updates"""

    def __init__(self,batch_size,num_features,d):
        WARPBatchUpdate.__init__(self,batch_size,d)
        self.dW = np.zeros((num_features,d))

    def clear(self):
        self.dW[:] = 0

    def set_update(self,ix,u,v_pos,v_neg,dU,dV_pos,dV_neg,dW):
        WARPBatchUpdate.set_update(self,ix,u,v_pos,v_neg,dU,dV_pos,dV_neg)
        self.dW += dW

class WARP2Decomposition(WARPDecomposition):

    def __init__(self,num_rows,num_cols,X,d):
        WARPDecomposition.__init__(self,num_rows,num_cols,d)
        # initialize factors to small random values
        self.U = d**-0.5*np.random.random_sample((num_rows,d))
        self.V = d**-0.5*np.random.random_sample((num_cols,d))
        # ensure memory layout avoids extra allocation in dot product
        self.U = np.asfortranarray(self.U)
        # W holds latent factors for each item feature
        self.W = d**-0.5*np.random.random_sample((X.shape[1],d))
        self.X = X

    def compute_update(self,updates,ix,u,i,j,L):
        # compute gradient update
        # j is the violating col i.e. U[u].V[j] is too large compared to U[u].V[i]
        dU = L*(self.V[i]-self.V[j])
        dV_pos = L*self.U[u]
        dV_neg = -L*self.U[u]
        dW = L*np.atleast_2d(self.X[i]-self.X[j]).T.dot(np.atleast_2d(self.U[u]))
        updates.set_update(ix,u,i,j,dU,dV_pos,dV_neg,dW)

    def apply_updates(self,updates,gamma,C):
        apply_updates(self.U,updates.u,updates.dU,gamma,C)
        apply_updates(self.V,updates.v_pos,updates.dV_pos,gamma,C)
        apply_updates(self.V,updates.v_neg,updates.dV_neg,gamma,C)
        self.apply_matrix_update(self.W,updates.dW,gamma,C)

    def apply_matrix_update(self,W,dW,gamma,C):
        W += gamma*dW
        # ensure that ||W_k|| < C for all k
        p = np.sum(np.abs(W)**2,axis=-1)**0.5/C
        p[p<1] = 1
        W /= p[:,np.newaxis]

    def reconstruct(self,rows):
        if rows is None:
            U = self.U
        else:
            U = np.asfortranarray(self.U[rows,:])
        return U.dot(self.V.T + self.X.dot(self.W).T)

class WARP2(WARP):
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

    def fit(self,train,X,validation=None):
        """
        Learn factors from training set. The dot product of the factors
        reconstructs the training matrix approximately, minimizing the
        WARP ranking loss relative to the original data.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            Training matrix to be factorized.
        X : numpy.ndarray.
            Item features.
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
        decomposition = WARP2Decomposition(num_rows,num_cols,X,self.d)
        updates = WARP2BatchUpdate(self.batch_size,X.shape[1],self.d)
        self.precompute_warp_loss(num_cols)

        self._fit(decomposition,updates,train,validation)

        self.U_ = decomposition.U
        self.V_ = decomposition.V
        self.W_ = decomposition.W

        return self

    def compute_updates(self,train,decomposition,updates):
        tot_trials = 0
        updates.clear()
        for ix in xrange(self.batch_size):
            u,i,j,N,trials = warp2_sample(decomposition.U,
                                          decomposition.V,
                                          decomposition.W,
                                          decomposition.X,
                                          train.data,
                                          train.indices,
                                          train.indptr,
                                          self.positive_thresh,
                                          self.max_trials)
            tot_trials += trials
            L = self.estimate_warp_loss(train,u,N)
            decomposition.compute_update(updates,ix,u,i,j,L)
        return tot_trials

