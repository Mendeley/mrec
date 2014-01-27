from itertools import izip
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.io import mmread, mmwrite
try:
    import cPickle as pickle
except ImportError:
    import pickle

from sparse import fast_sparse_matrix, loadtxt, loadz, savez
from base_recommender import BaseRecommender

__version__ = '0.3.1'

def load_fast_sparse_matrix(input_format,filepath):
    """
    Load a fast_sparse_matrix from an input file of the specified format,
    by delegating to the appropriate static method.

    Parameters
    ----------
    input_format : str
        Specifies the file format:
        - tsv
        - csv
        - mm  (MatrixMarket)
        - fsm (mrec.sparse.fast_sparse_matrix)
    filepath : str
        The file to load.
    """
    if input_format == 'tsv':
        return fast_sparse_matrix.loadtxt(filepath)
    elif input_format == 'csv':
        return fast_sparse_matrix.loadtxt(filepath,delimiter=',')
    elif input_format == 'mm':
        return fast_sparse_matrix.loadmm(filepath)
    elif input_format == 'fsm':
        return fast_sparse_matrix.load(filepath)
    raise ValueError('unknown input format: {0}'.format(input_format))

def load_sparse_matrix(input_format,filepath):
    """
    Load a scipy.sparse.csr_matrix from an input file of the specified format.

    Parameters
    ----------
    input_format : str
        Specifies the file format:
        - tsv
        - csv
        - mm  (MatrixMarket)
        - npz (scipy.sparse.csr_matrix serialized with mrec.sparse.savez())
        - fsm (mrec.sparse.fast_sparse_matrix)
    filepath : str
        The file to load.
    """
    if input_format == 'tsv':
        return loadtxt(filepath)
    elif input_format == 'csv':
        return loadtxt(filepath,delimiter=',')
    elif input_format == 'mm':
        return mmread(filepath).tocsr()
    elif input_format == 'npz':
        return loadz(filepath).tocsr()
    elif input_format == 'fsm':
        return fast_sparse_matrix.load(filepath).X
    raise ValueError('unknown input format: {0}'.format(input_format))

def save_sparse_matrix(data,fmt,filepath):
    """
    Save a scipy sparse matrix in the specified format. Row and column
    indices will be converted to 1-indexed if you specify a plain text
    format (tsv, csv, mm). Note that zero entries are guaranteed to be
    saved in tsv or csv format.

    Parameters
    ----------
    data : scipy sparse matrix to save
    fmt : str
        Specifies the file format to write:
        - tsv
        - csv
        - mm  (MatrixMarket)
        - npz (save as npz archive of numpy arrays)
        - fsm (mrec.sparse.fast_sparse_matrix)
    filepath : str
        The file to load.
    """
    if fmt == 'tsv':
        m = data.tocoo()
        with open(filepath,'w') as out:
            for u,i,v in izip(m.row,m.col,m.data):
                print >>out,'{0}\t{1}\t{2}'.format(u+1,i+1,v)
    elif fmt == 'csv':
        m = data.tocoo()
        with open(filepath,'w') as out:
            for u,i,v in izip(m.row,m.col,m.data):
                print >>out,'{0},{1},{2}'.format(u+1,i+1,v)
    elif fmt == 'mm':
        mmwrite(filepath,data)
    elif fmt == 'npz':
        savez(data.tocoo(),filepath)
    elif fmt == 'fsm':
        fast_sparse_matrix(data).save(filepath)
    else:
        raise ValueError('unknown output format: {0}'.format(fmt))

def save_recommender(model,filepath):
    """
    Save a recommender model to file.

    Parameters
    ----------
    model : mrec.base_recommender.BaseRecommender
        The recommender to save.
    filepath : str
        The filepath to write to.
    """
    model.save(filepath)

def load_recommender(filepath):
    """
    Load a recommender model from file after it has been saved by
    save_recommender().

    Parameters
    ----------
    filepath : str
        The filepath to read from.
    """
    return BaseRecommender.load(filepath)

def read_recommender_description(filepath):
    """
    Read a recommender model description from file after it has
    been saved by save_recommender(), without loading all the
    associated data into memory.

    Parameters
    ----------
    filepath : str
        The filepath to read from.
    """
    return BaseRecommender.read_recommender_description(filepath)
