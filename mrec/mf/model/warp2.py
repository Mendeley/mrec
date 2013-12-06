import numpy as np
import scipy
import random

from warp import WARPBatchUpdate, WARPDecomposition, WARP
from warp_fast import warp2_sample

class WARP2BatchUpdate(WARPBatchUpdate):
    """Collection of arrays to hold a batch of sgd updates."""

    def __init__(self,batch_size,num_features,d):
        WARPBatchUpdate.__init__(self,batch_size,d)
        self.dW = np.zeros((num_features,d))

    def clear(self):
        self.dW[:] = 0

    def set_update(self,ix,update):
        u,v_pos,v_neg,dU,dV_pos,dV_neg,dW = update
        WARPBatchUpdate.set_update(self,ix,(u,v_pos,v_neg,dU,dV_pos,dV_neg))
        self.dW += dW

class WARP2Decomposition(WARPDecomposition):
    """
    Joint matrix and feature embedding optimizing the WARP loss.

    Parameters
    ==========
    num_rows : int
        Number of rows in the full matrix.
    num_cols : int
        Number of columns in the full matrix.
    X : array_like, shape = [num_cols, num_features]
        Features describing each column in the matrix.
    d : int
        The embedding dimension.
    """

    def __init__(self,num_rows,num_cols,X,d):
        WARPDecomposition.__init__(self,num_rows,num_cols,d)
        # W holds latent factors for each item feature
        self.W = d**-0.5*np.random.random_sample((X.shape[1],d))
        self.X = X
        self.is_sparse = isinstance(X,scipy.sparse.csr_matrix)

    def compute_gradient_step(self,u,i,j,L):
        """
        Compute a gradient step from results of sampling.

        Parameters
        ==========
        u : int
            The sampled row.
        i : int
            The sampled positive column.
        j : int
            The sampled violating negative column i.e. U[u].V[j] is currently
            too large compared to U[u].V[i]
        L : int
            The number of trials required to find a violating negative column.

        Returns
        =======
        u : int
            As input.
        i : int
            As input.
        j : int
            As input.
        dU : numpy.ndarray
            Gradient step for U[u].
        dV_pos : numpy.ndarray
            Gradient step for V[i].
        dV_neg : numpy.ndarray
            Gradient step for V[j].
        dW : numpy.ndarray
            Gradient step for W.
        """
        dU = L*(self.V[i]-self.V[j])
        dV_pos = L*self.U[u]
        dV_neg = -L*self.U[u]
        dx = self.X[i]-self.X[j]
        if not self.is_sparse:
            dx = np.atleast_2d(dx)
        dW = L*dx.T.dot(np.atleast_2d(self.U[u]))
        return u,i,j,dU,dV_pos,dV_neg,dW

    def apply_updates(self,updates,gamma,C):
        WARPDecomposition.apply_updates(self,updates,gamma,C)
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
    W_ : numpy.ndarray
        Item feature factors.
    """

    def fit(self,train,X,validation=None):
        """
        Learn embedding from training set. A suitable dot product of the
        factors reconstructs the training matrix approximately, minimizing
        the WARP ranking loss relative to the original data.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            Training matrix to be factorized.
        X : array_like, shape = [num_cols, num_features]
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

    def sample(self,train,decomposition):
        # delegate to cython implementation
        return warp2_sample(decomposition.U,
                            decomposition.V,
                            decomposition.W,
                            decomposition.X,
                            train.data,
                            train.indices,
                            train.indptr,
                            self.positive_thresh,
                            self.max_trials)

