import random
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.utils.testing import assert_array_equal

def get_random_coo_matrix(rows=3,cols=10,nnz=20):
    row_col = random.sample(xrange(rows*cols),nnz)  # ensure <row,col> are unique
    row = [i // cols for i in row_col]
    col = [i % cols for i in row_col]
    data = np.random.randint(0,nnz*5,nnz)
    return coo_matrix((data,(row,col)),shape=(rows,cols))

def assert_sparse_matrix_equal(X,Y):
    expected = X.toarray()
    actual = Y.toarray()
    # it's possible that we had trailing empty columns in X
    # - there's no way we can know about these sometimes e.g.
    # when reading back from file
    expected = expected[:actual.shape[0],:actual.shape[1]]
    assert_array_equal(expected,actual)

