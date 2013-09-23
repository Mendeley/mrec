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

from mrec import load_fast_sparse_matrix, read_recommender_description
from mrec.parallel import predict

from filename_conventions import *

def process(view,opts,modelfile,trainfile,testfile,outdir,evaluator):

    logging.info('finding number of users...')
    dataset = load_fast_sparse_matrix(opts.input_format,trainfile)
    num_users,num_items = dataset.shape
    del dataset

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
    tasks = create_tasks(modelfile,opts.input_format,trainfile,opts.test_input_format,testfile,recsdir,num_users,opts.num_engines,done,evaluator)

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

def create_tasks(modelfile,input_format,trainfile,test_input_format,testfile,outdir,num_users,num_engines,done,evaluator):
    users_per_engine = int(math.ceil(float(num_users)/num_engines))
    tasks = []
    for start in xrange(0,num_users,users_per_engine):
        end = min(num_users,start+users_per_engine)
        generate = (start,end) not in done
        tasks.append((modelfile,input_format,trainfile,test_input_format,testfile,outdir,start,end,evaluator,generate))
    return tasks

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
    parser.add_option('-n','--num_engines',dest='num_engines',type='int',default=0,help='number of IPython engines to use')
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--test_input_format',dest='test_input_format',default='npz',help='format of test dataset(s) tsv | csv | mm (matrixmarket) | npz (numpy binary)  (default: %default)')
    parser.add_option('--train',dest='train',help='glob specifying path(s) to training dataset(s) IMPORTANT: must be in quotes if it includes the * wildcard')
    parser.add_option('--modeldir',dest='modeldir',help='directory containing trained models')
    parser.add_option('--outdir',dest='outdir',help='directory for output files')
    parser.add_option('--metrics',dest='metrics',default='main',help='which set of metrics to compute, main|hitrate (default: %default)')
    parser.add_option('--overwrite',dest='overwrite',action='store_true',default=False,help='overwrite existing files in outdir (default: %default)')
    parser.add_option('--packer',dest='packer',default='json',help='packer for IPython.parallel (default: %default)')
    parser.add_option('--add_module_paths',dest='add_module_paths',help='optional comma-separated list of paths to append to pythonpath (useful if you need to import uninstalled modules to IPython engines on a cluster)')

    metrics_funcs = {'main':compute_main_metrics,
                     'hitrate':compute_hit_rate}

    (opts,args) = parser.parse_args()
    if not opts.input_format or not opts.train or not opts.outdir or not opts.num_engines \
            or not opts.modeldir or opts.metrics not in metrics_funcs:
        parser.print_help()
        raise SystemExit

    opts.train = os.path.abspath(opts.train)
    opts.modeldir = os.path.abspath(opts.modeldir)
    opts.outdir = os.path.abspath(opts.outdir)

    # create an ipython client
    c = Client(packer=opts.packer)
    view = c.load_balanced_view()

    if opts.add_module_paths:
        c[:].execute('import sys')
        for path in opts.add_module_paths.split(','):
            logging.info('adding {0} to pythonpath on all engines'.format(path))
            c[:].execute("sys.path.append('{0}')".format(path))

    evaluator = Evaluator(metrics_funcs[opts.metrics],max_items=20)

    trainfiles = glob.glob(os.path.expanduser(opts.train))

    descriptions = set()
    all_metrics = defaultdict(list)
    for trainfile in trainfiles:
        logging.info('processing {0}...'.format(trainfile))
        modelfile = get_modelfile(trainfile,opts.modeldir)
        testfile = get_testfile(trainfile)
        description,metrics = process(view,opts,modelfile,trainfile,testfile,opts.outdir,evaluator)
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
