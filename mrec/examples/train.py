"""
Train an item similarity model in parallel on an ipython cluster.
We assume a shared filesystem (as you'll have when running locally
or on an AWS cluster fired up with StarCluster) to avoid passing
data between the controller and the worker engines, as this can
cause OOM issues for the controller.

You can specify multiple training sets and the model will learn a
separate similarity matrix for each input dataset: this makes it
easy to generate data for cross-validated evaluation.
"""

import math
import glob
import re
import os
import subprocess
from shutil import rmtree
import logging

from mrec import load_fast_sparse_matrix, save_recommender
from mrec.parallel import train

from filename_conventions import *

def process(view,opts,model,trainfile,outdir):

    logging.info('finding number of items...')
    dataset = load_fast_sparse_matrix(opts.input_format,trainfile)
    num_users,num_items = dataset.shape
    del dataset

    simsdir = get_simsdir(trainfile,outdir)
    logging.info('creating sims directory {0}...'.format(simsdir))
    subprocess.check_call(['mkdir','-p',simsdir])

    done = []
    if not opts.overwrite:
        logging.info('checking for existing output sims...')
        done.extend(find_done(simsdir))
        if done:
            logging.info('found {0} output files'.format(len(done)))

    logging.info('creating tasks...')
    tasks = create_tasks(model,opts.input_format,trainfile,simsdir,num_items,opts.num_engines,opts.max_sims,done)

    logging.info('running in parallel across ipython engines...')
    results = []
    results.append(view.map_async(train.run,tasks,retries=2))

    # wait for tasks to complete
    processed = [r.get() for r in results]

    logging.info('checking output files...')
    done = find_done(simsdir)
    remaining = len(tasks) - len(done)
    if remaining == 0:
        logging.info('SUCCESS: all tasks completed')
        logging.info('concatenating {0} partial output files...'.format(len(done)))
        paths = [os.path.join(simsdir,'sims.{0}-{1}.tsv'.format(start,end)) for start,end in done]
        cmd = ['cat']+paths
        simsfile = get_simsfile(trainfile,outdir)
        subprocess.check_call(cmd,stdout=open(simsfile,'w'))
        logging.info('removing partial output files...')
        rmtree(simsdir)
        model.load_similarity_matrix(simsfile,num_items)
        modelfile = get_modelfile(trainfile,outdir)
        save_recommender(model,modelfile)
        logging.info('done')
    else:
        logging.error('FAILED: {0}/{1} tasks did not complete successfully'.format(remaining,len(tasks)))
        logging.error('try rerunning the command to retry the remaining tasks')

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

def create_tasks(model,input_format,trainfile,outdir,num_items,num_engines,max_similar_items,done):
    items_per_engine = int(math.ceil(float(num_items)/num_engines))
    tasks = []
    for start in xrange(0,num_items,items_per_engine):
        end = min(num_items,start+items_per_engine)
        if (start,end) not in done:
            tasks.append((model,input_format,trainfile,outdir,start,end,max_similar_items))
    return tasks

def main():

    import os
    from optparse import OptionParser
    from IPython.parallel import Client

    from mrec.item_similarity.slim import SLIM
    from mrec.item_similarity.knn import CosineKNNRecommender, DotProductKNNRecommender
    from mrec.popularity import ItemPopularityRecommender

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('-n','--num_engines',dest='num_engines',type='int',default=0,help='number of IPython engines to use')
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--train',dest='train',help='glob specifying path(s) to training dataset(s) IMPORTANT: must be in quotes if it includes the * wildcard')
    parser.add_option('--outdir',dest='outdir',help='directory for output files')
    parser.add_option('--overwrite',dest='overwrite',action='store_true',help='overwrite existing files in outdir')
    parser.add_option('--model',dest='model',default='slim',help='type of model to train: slim | knn | popularity (default: %default)')
    parser.add_option('--max_sims',dest='max_sims',type='int',default=100,help='max similar items to output for each training item (default: %default)')
    parser.add_option('--learner',dest='learner',default='sgd',help='underlying learner for SLIM learner: sgd | elasticnet | fs_sgd (default: %default)')
    parser.add_option('--l1_reg',dest='l1_reg',type='float',default=0.1,help='l1 regularization constant (default: %default)')
    parser.add_option('--l2_reg',dest='l2_reg',type='float',default=0.001,help='l2 regularization constant (default: %default)')
    parser.add_option('--metric',dest='metric',default='cosine',help='metric for knn recommender: cosine | dot (default: %default)')
    parser.add_option('--popularity_method',dest='popularity_method',default='count',help='how to compute popularity for baseline recommender: count | sum | avg | thresh (default: %default)')
    parser.add_option('--popularity_thresh',dest='popularity_thresh',type='float',default=0,help='ignore scores below this when computing popularity for baseline recommender (default: %default)')
    parser.add_option('--packer',dest='packer',default='json',help='packer for IPython.parallel (default: %default)')
    parser.add_option('--add_module_paths',dest='add_module_paths',help='optional comma-separated list of paths to append to pythonpath (useful if you need to import uninstalled modules to IPython engines on a cluster)')

    (opts,args) = parser.parse_args()
    if not opts.input_format or not opts.train or not opts.outdir or not opts.num_engines:
        parser.print_help()
        raise SystemExit

    opts.train = os.path.abspath(opts.train)
    opts.outdir = os.path.abspath(opts.outdir)

    trainfiles = glob.glob(os.path.expanduser(opts.train))

    if opts.model == 'popularity':
        # special case, don't need to run in parallel
        subprocess.check_call(['mkdir','-p',opts.outdir])
        for trainfile in trainfiles:
            logging.info('processing {0}...'.format(trainfile))
            model = ItemPopularityRecommender(method=opts.popularity_method,thresh=opts.popularity_thresh)
            dataset = load_fast_sparse_matrix(opts.input_format,trainfile)
            model.train(dataset)
            modelfile = get_modelfile(trainfile,opts.outdir)
            save_recommender(model,modelfile)
        logging.info('done')
        return

    # create an ipython client
    c = Client(packer=opts.packer)
    view = c.load_balanced_view()

    if opts.add_module_paths:
        c[:].execute('import sys')
        for path in opts.add_module_paths.split(','):
            logging.info('adding {0} to pythonpath on all engines'.format(path))
            c[:].execute("sys.path.append('{0}')".format(path))

    if opts.model == 'slim':
        if opts.learner == 'fs_sgd':
            num_selected_features = 2 * opts.max_sims  # preselect this many candidate similar items
            model = SLIM(l1_reg=opts.l1_reg,l2_reg=opts.l2_reg,model=opts.learner,num_selected_features=num_selected_features)
        else:
            model = SLIM(l1_reg=opts.l1_reg,l2_reg=opts.l2_reg,model=opts.learner)
    elif opts.model == 'knn':
        if opts.metric == 'cosine':
            model = CosineKNNRecommender(k=opts.max_sims)
        elif opts.metric == 'dot':
            model = DotProductKNNRecommender(k=opts.max_sims)
        else:
            parser.print_help()
            raise SystemExit('unknown metric: {0}'.format(opts.metric))
    else:
        parser.print_help()
        raise SystemExit('unknown model type: {0}'.format(opts.model))

    for trainfile in trainfiles:
        logging.info('processing {0}...'.format(trainfile))
        process(view,opts,model,trainfile,opts.outdir)

if __name__ == '__main__':
    main()
