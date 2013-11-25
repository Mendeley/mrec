import tempfile
import os

from mrec.testing import get_random_coo_matrix
from mrec.testing import assert_sparse_matrix_equal

from mrec import load_sparse_matrix
from mrec import save_sparse_matrix

def test_save_load_sparse_matrix():
    X = get_random_coo_matrix()
    for fmt in ['tsv','csv','npz','mm','fsm']:
        if fmt == 'mm':
            suffix = '.mtx'
        elif fmt == 'npz' or fmt == 'fsm':
            suffix = '.npz'
        else:
            suffix = ''
        f,path = tempfile.mkstemp(suffix=suffix)
        save_sparse_matrix(X,fmt,path)
        Y = load_sparse_matrix(fmt,path)
        assert_sparse_matrix_equal(X,Y)
        os.remove(path)
