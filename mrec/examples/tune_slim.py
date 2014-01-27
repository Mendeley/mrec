"""
Try to find a sensible range for regularization
constants for SLIM by looking at model sparsity.
"""

import random
from math import log10
import logging
from operator import itemgetter
from optparse import OptionParser
try:
    from sklearn.grid_search import ParameterGrid
except ImportError:
    from sklearn.grid_search import IterGrid as ParameterGrid
from IPython.parallel import Client

from mrec import load_fast_sparse_matrix

def estimate_sparsity(task):
    from mrec.item_similarity.slim import SLIM
    args,dataset,min_nnz,sample_items = task
    model = SLIM(**args)
    tot_nnz = 0
    tot_neg = 0
    below_min_nnz = 0

    for i in sample_items:
        w = model.compute_similarities(dataset,i)
        nnz = sum(w>0)
        tot_nnz += nnz
        if nnz < min_nnz:
            below_min_nnz += 1
        tot_neg += sum(w<0)

    num_samples = len(sample_items)
    avg_nnz = float(tot_nnz)/num_samples
    too_few_sims = float(below_min_nnz)/num_samples
    avg_neg = float(tot_neg)/num_samples
    return args,avg_nnz,too_few_sims,avg_neg

def pow_range(small,big):
    return [10**v for v in xrange(int(log10(small)),int(log10(big))+1)]

def main():
    parser = OptionParser()
    parser.add_option('-d','--dataset',dest='dataset',help='path to dataset')
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--l1_min',dest='l1_min',type='float',help='min l1 constant to try (expected to be a power of 10)')
    parser.add_option('--l1_max',dest='l1_max',type='float',help='max l1 constant to try (expected to be a power of 10)')
    parser.add_option('--l2_min',dest='l2_min',type='float',help='min l2 constant to try (expected to be a power of 10)')
    parser.add_option('--l2_max',dest='l2_max',type='float',help='max l2 constant to try (expected to be a power of 10)')
    parser.add_option('--max_sims',dest='max_sims',type='int',default=2000,help='max desired number of positive item similarity weights (default: %default)')
    parser.add_option('--min_sims',dest='min_sims',type='int',default=15,help='min desired number of positive item similarity weights (default: %default)')
    parser.add_option('--max_sparse',dest='max_sparse',type='float',default=0.01,help='max allowable proportion of items with less than min_sims positive similarity weights (default: %default)')
    parser.add_option('--num_samples',dest='num_samples',type='int',default=100,help='number of sample items to evaluate for each regularization setting')
    parser.add_option('--packer',dest='packer',default='json',help='packer for IPython.parallel (default: %default)')
    parser.add_option('--add_module_paths',dest='add_module_paths',help='comma-separated list of paths to append to pythonpath to enable import of uninstalled modules')

    (opts,args) = parser.parse_args()
    if not opts.dataset or not opts.input_format or not opts.l1_min or not opts.l1_max or not opts.l2_min or not opts.l2_max:
        parser.print_help()
        raise SystemExit

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    dataset = load_fast_sparse_matrix(opts.input_format,opts.dataset)

    params = {'l1_reg':pow_range(opts.l1_min,opts.l1_max),
              'l2_reg':pow_range(opts.l2_min,opts.l2_max)}
    num_items = dataset.shape[1]
    sample_items = random.sample(xrange(num_items),opts.num_samples)

    logging.info('preparing tasks for a grid search of these values:')
    logging.info(params)
    tasks = [(args,dataset,opts.min_sims,sample_items) for args in ParameterGrid(params)]

    c = Client(packer=opts.packer)
    view = c.load_balanced_view()

    if opts.add_module_paths:
        c[:].execute('import sys')
        for path in opts.add_module_paths.split(','):
            logging.info('adding {0} to pythonpath on all engines'.format(path))
            c[:].execute("sys.path.append('{0}')".format(path))

    logging.info('running {0} tasks in parallel...'.format(len(tasks)))
    results = view.map(estimate_sparsity,tasks,ordered=False)

    candidates = [(args,nsims,nsparse,nneg) for args,nsims,nsparse,nneg in results if nsims <= opts.max_sims and nsparse <= opts.max_sparse]

    if candidates:
        best = min(candidates,key=itemgetter(1))

        print 'best parameter setting: {0}'.format(best[0])
        print 'mean # positive similarity weights per item = {0:.3}'.format(best[1])
        print 'proportion of items with fewer than {0} positive similarity weights = {1:.3}'.format(opts.min_sims,best[2])
        print 'mean # negative similarity weights per item = {0:.3}'.format(best[3])
    else:
        print 'no parameter settings satisfied the conditions, try increasing --min_sims, --max_sims or --max_sparse'

if __name__ == '__main__':
    main()
