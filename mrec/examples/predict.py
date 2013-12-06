"""
Make and evaluate recommendations in parallel on an ipython cluster,
using models that have previously been trained and saved to file.
We assume a shared filesystem (as you'll have when running locally
or on an AWS cluster fired up with StarCluster) to avoid passing
data between the controller and the worker engines, as this can
cause OOM issues for the controller.

You can specify multiple training sets / models and separate
recommendations will be output and evaluated for each of them: this
makes it easy to run a cross-validated evaluation.
"""

import math
import glob
import re
import os
import subprocess
from shutil import rmtree
import logging
from collections import defaultdict

from mrec import load_sparse_matrix, read_recommender_description, load_recommender
from mrec.parallel import predict
from mrec.mf.recommender import MatrixFactorizationRecommender
from mrec.item_similarity.recommender import ItemSimilarityRecommender

from filename_conventions import *

ONE_MB = 2**20

def process(view,opts,modelfile,trainfile,testfile,featurefile,outdir,evaluator):

    recsdir = get_recsdir(trainfile,opts.outdir)
    logging.info('creating recs directory {0}...'.format(recsdir))
    subprocess.check_call(['mkdir','-p',recsdir])

    done = []
    if not opts.overwrite:
        logging.info('checking for existing output recs...')
        done.extend(find_done(recsdir))
        if done:
            logging.info('found {0} output files'.format(len(done)))

    logging.info('creating tasks...')
    tasks = create_tasks(modelfile,
                         opts.input_format,
                         trainfile,
                         opts.test_input_format,
                         testfile,
                         opts.item_feature_format,
                         featurefile,
                         recsdir,
                         opts.mb_per_task,
                         done,
                         evaluator)

    logging.info('running in parallel across ipython engines...')
    results = []
    results.append(view.map_async(predict.run,tasks,retries=2))

    # wait for tasks to complete
    processed = [r.get() for r in results]

    logging.info('checking output files...')
    done = find_done(recsdir)
    remaining = len(tasks) - len(done)

    if remaining == 0:
        logging.info('SUCCESS: all tasks completed')
        logging.info('concatenating {0} partial output files...'.format(len(done)))
        paths = [os.path.join(recsdir,'recs.{0}-{1}.tsv'.format(start,end)) for start,end in done]
        cmd = ['cat']+paths
        recsfile = get_recsfile(trainfile,outdir)
        subprocess.check_call(cmd,stdout=open(recsfile,'w'))
        logging.info('removing partial output files...')
        rmtree(recsdir)
        logging.info('done')

        # aggregate metrics from each task
        avg_metrics = defaultdict(float)
        tot_count = 0
        for results in processed:
            for cum_metrics,count in results:
                for m,val in cum_metrics.iteritems():
                    avg_metrics[m] += val
                tot_count += count
        for m in avg_metrics:
            avg_metrics[m] /= float(tot_count)
    else:
        logging.error('FAILED: {0}/{1} tasks did not complete successfully'.format(remaining,len(tasks)))
        logging.error('try rerunning the command to retry the remaining tasks')
        avg_metrics = None

    return read_recommender_description(modelfile),avg_metrics

def create_tasks(modelfile,
                 input_format,
                 trainfile,
                 test_input_format,
                 testfile,
                 item_feature_format,
                 featurefile,
                 outdir,
                 mb_per_task,
                 done,
                 evaluator):
    users_per_task,num_users = estimate_users_per_task(mb_per_task,input_format,trainfile,modelfile)
    tasks = []
    for start in xrange(0,num_users,users_per_task):
        end = min(num_users,start+users_per_task)
        generate = (start,end) not in done
        tasks.append((modelfile,input_format,trainfile,test_input_format,testfile,item_feature_format,featurefile,outdir,start,end,evaluator,generate))
    logging.info('created {0} tasks, {1} users per task'.format(len(tasks),users_per_task))
    return tasks

def estimate_users_per_task(mb_per_task,input_format,trainfile,modelfile):
    num_users,num_items,nnz = get_dataset_size(input_format,trainfile)
    logging.info('loading model to get size...')
    model = load_recommender(modelfile)
    # we load the training and test data on every task
    # - let's guess that worst case the test data will be the same size
    required_mb_per_task = 2*(nnz*16)/ONE_MB
    if isinstance(model,MatrixFactorizationRecommender):
        # we have to load the factors on every task
        required_mb_per_task += ((model.U.size+model.V.size)*16)/ONE_MB
        if mb_per_task > required_mb_per_task:
            # remaining mem usage is dominated by computed scores:
            users_per_task = ((mb_per_task-required_mb_per_task)*ONE_MB) / (num_items*16)
    elif isinstance(model,ItemSimilarityRecommender):
        # we have to load the similarity matrix on every task
        required_mb_per_task += (model.similarity_matrix.nnz*16)/ONE_MB
        if mb_per_task > required_mb_per_task:
            # estimate additional usage from avg items per user and sims per item
            items_per_user = nnz / num_users
            sims_per_item = model.similarity_matrix.nnz / num_items
            users_per_task = ((mb_per_task-required_mb_per_task)*ONE_MB) / (items_per_user*sims_per_item*16)
    else:
        # assume nothing else to load
        users_per_task = num_users

    if mb_per_task <= required_mb_per_task:
        raise RuntimeError('requires at least {0}MB per task, increase --mb_per_task if you can'.format(required_mb_per_task))

    return users_per_task,num_users

def get_dataset_size(input_format,datafile):
    logging.info('loading dataset to get size...')
    dataset = load_sparse_matrix(input_format,datafile)
    return dataset.shape[0],dataset.shape[1],dataset.nnz

def find_done(outdir):
    success_files = glob.glob(os.path.join(outdir,'*.SUCCESS'))
    r = re.compile('.*?([0-9]+)-([0-9]+)\.SUCCESS$')
    done = []
    for path in success_files:
        m = r.match(path)
        start = int(m.group(1))
        end = int(m.group(2))
        done.append((start,end))
    return done

def main():

    import os
    from optparse import OptionParser
    from IPython.parallel import Client

    from mrec.evaluation.metrics import compute_main_metrics, compute_hit_rate
    from mrec.evaluation import Evaluator
    from mrec import load_recommender
    from mrec.evaluation.metrics import print_report

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('--mb_per_task',dest='mb_per_task',type='int',default=None,help='approximate memory limit per task in MB, so total memory usage is num_engines * mb_per_task (default: share all available RAM across engines)')
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--test_input_format',dest='test_input_format',default='npz',help='format of test dataset(s) tsv | csv | mm (matrixmarket) | npz (numpy binary)  (default: %default)')
    parser.add_option('--train',dest='train',help='glob specifying path(s) to training dataset(s) IMPORTANT: must be in quotes if it includes the * wildcard')
    parser.add_option('--modeldir',dest='modeldir',help='directory containing trained models')
    parser.add_option('--outdir',dest='outdir',help='directory for output files')
    parser.add_option('--metrics',dest='metrics',default='main',help='which set of metrics to compute, main|hitrate (default: %default)')
    parser.add_option('--item_feature_format',dest='item_feature_format',help='format of item features tsv | csv | mm (matrixmarket) | npz (numpy arrays)')
    parser.add_option('--item_features',dest='item_features',help='path to sparse item features in tsv format (item_id,feature_id,val)')
    parser.add_option('--overwrite',dest='overwrite',action='store_true',default=False,help='overwrite existing files in outdir (default: %default)')
    parser.add_option('--packer',dest='packer',default='json',help='packer for IPython.parallel (default: %default)')
    parser.add_option('--add_module_paths',dest='add_module_paths',help='optional comma-separated list of paths to append to pythonpath (useful if you need to import uninstalled modules to IPython engines on a cluster)')

    metrics_funcs = {'main':compute_main_metrics,
                     'hitrate':compute_hit_rate}

    (opts,args) = parser.parse_args()
    if not opts.input_format or not opts.train or not opts.outdir \
            or not opts.modeldir or opts.metrics not in metrics_funcs:
        parser.print_help()
        raise SystemExit

    opts.train = os.path.abspath(os.path.expanduser(opts.train))
    opts.modeldir = os.path.abspath(os.path.expanduser(opts.modeldir))
    opts.outdir = os.path.abspath(os.path.expanduser(opts.outdir))

    # create an ipython client
    c = Client(packer=opts.packer)
    view = c.load_balanced_view()
    if opts.mb_per_task is None:
        import psutil
        num_engines = len(view)
        opts.mb_per_task = psutil.virtual_memory().available/ONE_MB/(num_engines+1)  # don't take *all* the memory

    if opts.add_module_paths:
        c[:].execute('import sys')
        for path in opts.add_module_paths.split(','):
            logging.info('adding {0} to pythonpath on all engines'.format(path))
            c[:].execute("sys.path.append('{0}')".format(path))

    evaluator = Evaluator(metrics_funcs[opts.metrics],max_items=20)

    trainfiles = glob.glob(opts.train)

    descriptions = set()
    all_metrics = defaultdict(list)
    for trainfile in trainfiles:
        logging.info('processing {0}...'.format(trainfile))
        modelfile = get_modelfile(trainfile,opts.modeldir)
        testfile = get_testfile(trainfile)
        description,metrics = process(view,opts,modelfile,trainfile,testfile,opts.item_features,opts.outdir,evaluator)
        descriptions.add(description)
        if metrics is not None:
            for m in metrics:
                all_metrics[m].append(metrics[m])

    description = ' AND '.join(descriptions)
    if len(descriptions) > 1:
        logging.warn('You are aggregating metrics from different models! {}'.format(description))

    print_report([description],[all_metrics])

if __name__ == '__main__':
    main()
