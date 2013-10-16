"""
Sparse data structures and convenience methods to load sparse matrices from file.
"""

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from scipy.io import mmread

def loadtxt(filepath,comments='#',delimiter=None,skiprows=0,usecols=None,index_offset=1):
    """
    Load a scipy sparse matrix from simply formatted data such as TSV, handles
    similar input to numpy.loadtxt().

    Parameters
    ----------

    filepath : file or str
        File containing simply formatted row,col,val sparse matrix data.
    comments : str, optional
        The character used to indicate the start of a comment (default: #).
    delimiter : str, optional
        The string used to separate values. By default, this is any whitespace.
    skiprows : int, optional
        Skip the first skiprows lines; default: 0.
    usecols : sequence, optional
        Which columns to read, with 0 being the first. For example, usecols = (1,4,5)
        will extract the 2nd, 5th and 6th columns. The default, None, results in all
        columns being read.
    index_offset : int, optional
        Offset applied to the row and col indices in the input data (default: 1).
        The default offset is chosen so that 1-indexed data on file results in a
        fast_sparse_matrix holding 0-indexed matrices.

    Returns
    -------

    mat : scipy.sparse.coo_matrix
        The sparse matrix.
    """
    d = np.loadtxt(filepath,comments=comments,delimiter=delimiter,skiprows=skiprows,usecols=usecols)
    if d.shape[1] < 3:
        raise ValueError('invalid number of columns in input')
    row = d[:,0]-index_offset
    col = d[:,1]-index_offset
    data = d[:,2]
    shape = (max(row)+1,max(col)+1)
    return coo_matrix((data,(row,col)),shape=shape)

def savez(d,filepath):
    """
    Save a sparse matrix to file in numpy binary format.

    Parameters
    ----------
    d : scipy sparse matrix
        The sparse matrix to save.
    filepath : str
        The filepath to write to.
    """
    np.savez(filepath,row=d.row,col=d.col,data=d.data,shape=d.shape)

def loadz(filepath):
    """
    Load a sparse matrix saved to file with savez.

    Parameters
    ----------
    filepath : str
        The filepath to read from.
    """
    y = np.load(filepath)
    return coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])

class fast_sparse_matrix(object):
    """
    Adds fast columnar reads and updates to
    a scipy.sparse.csr_matrix, at the cost
    of keeping a csc_matrix of equal size as
    a column-wise index into the same raw data.
    It is updateable in the sense that you can
    change the values of all the existing non-
    zero entries in a given column.  Trying to
    set other entries will result in an error.

    For other functionality you are expected to
    call methods on the underlying csr_matrix:

        fsm = fast_sparse_matrix(data) # data is a csr_matrix
        col = fsm.fast_get_col(2)      # get a column quickly
        row = fsm.X[1]                 # get a row as usual
    """
    def __init__(self,X,col_view=None):
        """
        Create a fast_sparse_matrix from a csr_matrix X.

        Parameters
        ----------

        X : scipy sparse matrix
            The sparse matrix to wrap.
        col_view : scipy.csc_matrix, optional
            The corresponding index matrix to provide fast columnar access,
            created if not supplied here.
        """
        self.X = X.tocsr()
        if col_view is not None:
            self.col_view = col_view
        else:
            # create the columnar index matrix
            ind = self.X.copy()
            ind.data = np.arange(self.X.nnz)
            self.col_view = ind.tocsc()

    @property
    def shape(self):
        """Return the shape of the underlying matrix."""
        return self.X.shape

    def fast_get_col(self,j):
        """Return column j."""
        col = self.col_view[:,j].copy()
        col.data = self.X.data[col.data]
        return col

    def fast_update_col(self,j,vals):
        """Update values of existing non-zeros in column j."""
        dataptr = self.col_view[:,j].data
        self.X.data[dataptr] = vals

    def save(self,filepath):
        """Save to file as arrays in numpy binary format."""
        d = self.X.tocoo(copy=False)
        v = self.col_view.tocoo(copy=False)
        np.savez(filepath,row=d.row,col=d.col,data=d.data,shape=d.shape,
                 v_row=v.row,v_col=v.col,v_data=v.data,v_shape=v.shape)

    @staticmethod
    def load(filepath):
        """
        Load a fast_sparse_matrix from file written by fast_sparse_matrix.save().
        """
        y = np.load(filepath,mmap_mode='r')
        X = coo_matrix((y['data'],(y['row'],y['col'])),shape=y['shape'])
        col_view = coo_matrix((y['v_data'],(y['v_row'],y['v_col'])),shape=y['v_shape'])
        return fast_sparse_matrix(X,col_view.tocsc())

    @staticmethod
    def loadtxt(filepath,comments='#',delimiter=None,skiprows=0,usecols=None,index_offset=1):
        """
        Create a fast_sparse_matrix from simply formatted data such as TSV, handles
        similar input to numpy.loadtxt().

        Parameters
        ----------

        filepath : file or str
            File containing simply formatted row,col,val sparse matrix data.
        comments : str, optional
            The character used to indicate the start of a comment (default: #).
        delimiter : str, optional
            The string used to separate values. By default, this is any whitespace.
        skiprows : int, optional
            Skip the first skiprows lines; default: 0.
        usecols : sequence, optional
            Which columns to read, with 0 being the first. For example, usecols = (1,4,5)
            will extract the 2nd, 5th and 6th columns. The default, None, results in all
            columns being read.
        index_offset : int, optional
            Offset applied to the row and col indices in the input data (default: 1).
            The default offset is chosen so that 1-indexed data on file results in a
            fast_sparse_matrix holding 0-indexed matrices.
        """
        X = loadtxt(filepath,comments=comments,delimiter=delimiter,skiprows=skiprows,usecols=usecols)
        return fast_sparse_matrix(X)

    @staticmethod
    def loadmm(filepath):
        """
        Create a fast_sparse_matrix from matrixmarket data.

        Parameters
        ----------

        filepath : file or str
            The matrixmarket file to read.
        """
        X = mmread(filepath)
        return fast_sparse_matrix(X)

