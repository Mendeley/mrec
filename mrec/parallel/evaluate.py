"""
Evaluation task to run on an ipython engine.
"""


def run(task):
    # import modules required by engine

    from mrec import load_sparse_matrix

    input_format, testfile, recsfile, start, end, evaluator = task

    # load the test data
    testdata = load_sparse_matrix(input_format, testfile)

    return evaluator.process(testdata, recsfile, start, end)
