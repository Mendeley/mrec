import tempfile
import os
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal

from mrec.testing import get_random_coo_matrix
from mrec.testing import assert_sparse_matrix_equal

from mrec.sparse import loadtxt
from mrec.sparse import savez
from mrec.sparse import loadz
from mrec.sparse import fast_sparse_matrix

def test_loadtxt():
    X = get_random_coo_matrix()
    f,path = tempfile.mkstemp(suffix='.npz')
    with open(path,'w') as f:
        for i,j,v in zip(X.row,X.col,X.data):
            print >>f,'{0}\t{1}\t{2}'.format(i+1,j+1,v)
    Y = loadtxt(path)
    os.remove(path)
    assert_sparse_matrix_equal(X,Y)

def test_savez_loadz():
    m = get_random_coo_matrix()
    f,path = tempfile.mkstemp(suffix='.npz')
    savez(m,path)
    n = loadz(path)
    os.remove(path)
    assert_array_equal(n.toarray(),m.toarray())

def test_init_fast_sparse_matrix():
    X = get_random_coo_matrix()
    Y = X.tocsr()
    Z = X.tocsc()
    for M in [X,Y,Z]:
        m = fast_sparse_matrix(M)
        assert_array_equal(m.X.toarray(),M.toarray())
        assert_equal(m.shape,M.shape)

def test_fast_get_col():
    X = get_random_coo_matrix().tocsc()
    m = fast_sparse_matrix(X)
    rows,cols = X.shape
    for j in xrange(cols):
        assert_array_equal(m.fast_get_col(j).toarray(),X[:,j].toarray())

def test_fast_update_col():
    X = get_random_coo_matrix().tocsc()
    m = fast_sparse_matrix(X)
    cols = X.shape[1]
    for j in xrange(cols):
        vals = m.fast_get_col(j).data
        if (vals==0).all():
            continue
        vals[vals!=0] += 1
        m.fast_update_col(j,vals)
        expected = X[:,j].toarray()
        for i in xrange(expected.shape[0]):
            if expected[i] != 0:
                expected[i] += 1
        assert_array_equal(m.fast_get_col(j).toarray(),expected)

def test_save_load():
    """Save to file as arrays in numpy binary format."""
    X = get_random_coo_matrix()
    m = fast_sparse_matrix(X)
    f,path = tempfile.mkstemp(suffix='.npz')
    m.save(path)
    n = fast_sparse_matrix.load(path)
    os.remove(path)
    assert_equal(m.shape,n.shape)
    assert_array_equal(m.X.toarray(),n.X.toarray())
    assert_array_equal(m.col_view.toarray(),n.col_view.toarray())

