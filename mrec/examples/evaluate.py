"""
Evaluate precomputed recommendations for one or more training/test sets.
Test and recommendation files must following naming conventions relative
to the training filepaths.
"""

def main():

    import os
    import logging
    import glob
    from optparse import OptionParser
    from collections import defaultdict

    from mrec import load_sparse_matrix
    from mrec.evaluation.metrics import compute_main_metrics, compute_hit_rate
    from mrec.evaluation import Evaluator
    from mrec.evaluation.metrics import print_report
    from filename_conventions import get_testfile, get_recsfile

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--test_input_format',dest='test_input_format',default='npz',help='format of test dataset(s) tsv | csv | mm (matrixmarket) | npz (numpy binary)  (default: %default)')
    parser.add_option('--train',dest='train',help='glob specifying path(s) to training dataset(s) IMPORTANT: must be in quotes if it includes the * wildcard')
    parser.add_option('--recsdir',dest='recsdir',help='directory containing tsv files of precomputed recommendations')
    parser.add_option('--metrics',dest='metrics',default='main',help='which set of metrics to compute, main|hitrate (default: %default)')
    parser.add_option('--description',dest='description',help='description of model which generated the recommendations')
    metrics_funcs = {'main':compute_main_metrics,
                     'hitrate':compute_hit_rate}

    (opts,args) = parser.parse_args()
    if not opts.input_format or not opts.train or not opts.recsdir \
            or opts.metrics not in metrics_funcs:
        parser.print_help()
        raise SystemExit

    opts.train = os.path.abspath(os.path.expanduser(opts.train))
    opts.recsdir = os.path.abspath(os.path.expanduser(opts.recsdir))

    evaluator = Evaluator(metrics_funcs[opts.metrics],max_items=20)

    trainfiles = glob.glob(opts.train)

    all_metrics = defaultdict(list)
    for trainfile in trainfiles:
        logging.info('processing {0}...'.format(trainfile))
        testfile = get_testfile(trainfile)
        recsfile = get_recsfile(trainfile,opts.recsdir)
        testdata = load_sparse_matrix(opts.test_input_format,testfile).tocsr()
        cum_metrics,count = evaluator.process(testdata,recsfile,0,testdata.shape[0])
        if cum_metrics is not None:
            for m in cum_metrics:
                all_metrics[m].append(float(cum_metrics[m])/count)

    print_report([opts.description],[all_metrics])

if __name__ == '__main__':
    main()
