import tempfile
import os

from mrec.testing import get_random_coo_matrix
from mrec.testing import assert_sparse_matrix_equal

from mrec import load_sparse_matrix
from mrec import save_sparse_matrix
from mrec import load_recommender
from mrec import save_recommender
from mrec import read_recommender_description

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

def test_save_load_recommender():
    """
    Save a recommender model to file.  If the model holds similarity matrix
    then numpy.savez is used to save it to disk efficiently, otherwise the
    model is simply pickled.

    Parameters
    ----------
    model : mrec.base_recommender.BaseRecommender
        The recommender to save.
    filepath : str
        The filepath to write to.
    """
    # TODO: test that an item similarity recommender saves and reconstructs
    # its similarity matrix
    # TODO: likewise for U,V factors for an mf recommender
    pass

def test_read_recommender_description():
    """
    Read a recommender model description from file after it has
    been saved by save_recommender(), without loading all the
    associated data into memory.

    Parameters
    ----------
    filepath : str
        The filepath to read from.
    """
    pass
