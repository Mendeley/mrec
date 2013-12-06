"""
Sparse data structures and convenience methods to load sparse matrices from file.
"""

import random
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
    mat : scipy.sparse.csr_matrix
        The sparse matrix.
    """
    d = np.loadtxt(filepath,comments=comments,delimiter=delimiter,skiprows=skiprows,usecols=usecols)
    if d.shape[1] < 3:
        raise ValueError('invalid number of columns in input')
    row = d[:,0]-index_offset
    col = d[:,1]-index_offset
    data = d[:,2]
    shape = (max(row)+1,max(col)+1)
    return csr_matrix((data,(row,col)),shape=shape)

def savez(d,file):
    """
    Save a sparse matrix to file in numpy binary format.

    Parameters
    ----------
    d : scipy.sparse.coo_matrix
        The sparse matrix to save.
    file : str or file
        Either the file name (string) or an open file (file-like object)
        where the matrix will be saved. If file is a string, the ``.npz``
        extension will be appended to the file name if it is not already there.
    """
    np.savez(file,row=d.row,col=d.col,data=d.data,shape=d.shape)

def loadz(file):
    """
    Load a sparse matrix saved to file with savez.

    Parameters
    ----------
    file : str
        The open file or filepath to read from.

    Returns
    -------
    mat : scipy.sparse.coo_matrix
        The sparse matrix.
    """
    y = np.load(file)
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

    >>> fsm = fast_sparse_matrix(data) # data is a csr_matrix
    >>> col = fsm.fast_get_col(2)      # get a column quickly
    >>> row = fsm.X[1]                 # get a row as usual
    """
    def __init__(self,X,col_view=None):
        """
        Create a fast_sparse_matrix from a csr_matrix X. Note
        that X is not copied and its values will be modified by
        any subsequent call to fast_update_col().

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
        """
        Return the shape of the underlying matrix.
        """
        return self.X.shape

    def fast_get_col(self,j):
        """
        Return column j of the underlying matrix.

        Parameters
        ----------
        j : int
            Index of column to get.

        Returns
        -------
        col : scipy.sparse.csc_matrix
            Copy of column j of the matrix.
        """
        col = self.col_view[:,j].copy()
        col.data = self.X.data[col.data]
        return col

    def fast_update_col(self,j,vals):
        """
        Update values of existing non-zeros in column
        of the underlying matrix.

        Parameters
        ----------
        j : int
            Index of the column to update.
        vals : array like
            The new values to be assigned, must satisfy
            len(vals) == X[:,j].nnz i.e. this method can
            only change the value of existing non-zero entries
            of column j, it cannot add new ones.
        """
        dataptr = self.col_view[:,j].data
        self.X.data[dataptr] = vals

    def ensure_sparse_cols(self,max_density,remove_lowest=True):
        """
        Ensure that no column of the matrix excess the specified
        density, setting excess entries to zero where necessary.

        This can be useful to avoid popularity bias in collaborative
        filtering, by pruning the number of users for popular items:

        >>> num_users,num_items = train.shape
        >>> f = fast_sparse_matrix(train)
        >>> f.ensure_sparse_cols(max_density=0.01)

        Now any item in train has non-zero ratings from at most 1% of users.

        Parameters
        ----------
        max_density : float
            The highest allowable column-wise density. A value of one
            or more is treated as an absolute limit on the number of
            non-zero entries in a column, while a value of less than
            one is treated as a density i.e. a proportion of the overall
            number of rows.
        remove_lowest : boolean (default: True)
            If true then excess entries to be set to zero in a column are
            chosen lowest first, otherwise they are selected randomly.
        """
        if max_density >= 1:
            max_nnz = int(max_density)
        else:
            max_nnz = int(max_density*self.shape[0])
        for j in xrange(self.shape[1]):
            col = self.fast_get_col(j)
            excess = col.nnz - max_nnz
            if excess > 0:
                if remove_lowest:
                    zero_entries = np.argsort(col.data)[:excess]
                else:
                    zero_entries = random.sample(xrange(col.nnz),excess)
                col.data[zero_entries] = 0
                self.fast_update_col(j,col.data)

    def save(self,filepath):
        """
        Save to file as arrays in numpy binary format.

        Parameters
        ----------
        filepath : str
            The filepath to write to.
        """
        d = self.X.tocoo(copy=False)
        v = self.col_view.tocoo(copy=False)
        np.savez(filepath,row=d.row,col=d.col,data=d.data,shape=d.shape,
                 v_row=v.row,v_col=v.col,v_data=v.data,v_shape=v.shape)

    @staticmethod
    def load(filepath):
        """
        Load a fast_sparse_matrix from file written by fast_sparse_matrix.save().

        Parameters
        ----------
        filepath : str
            The filepath to load.
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

        Returns
        -------
        mat : mrec.sparse.fast_sparse_matrix
            A fast_sparse_matrix holding the data in the file.
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

        Returns
        -------
        mat : mrec.sparse.fast_sparse_matrix
            A fast_sparse_matrix holding the data in the file.
        """
        X = mmread(filepath)
        return fast_sparse_matrix(X)

