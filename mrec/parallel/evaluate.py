"""
Evaluation task to run on an ipython engine.
"""

def run(task):

    # import modules required by engine
    import numpy as np
    from scipy.sparse import coo_matrix
    from collections import defaultdict

    from mrec import load_sparse_matrix

    input_format,testfile,recsfile,start,end,evaluator = task

    # load the test data
    testdata = load_sparse_matrix(input_format,testfile)

    return evaluator.process(testdata,recsfile,start,end)
