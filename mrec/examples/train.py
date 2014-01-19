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

from filename_conventions import *

def main():

    import os
    import logging
    import glob
    import subprocess
    from optparse import OptionParser
    from IPython.parallel import Client

    from mrec import load_fast_sparse_matrix, save_recommender
    from mrec.item_similarity.slim import SLIM
    from mrec.item_similarity.knn import CosineKNNRecommender, DotProductKNNRecommender
    from mrec.mf.wrmf import WRMFRecommender
    from mrec.mf.warp import WARPMFRecommender
    from mrec.mf.warp2 import WARP2MFRecommender
    from mrec.popularity import ItemPopularityRecommender
    from mrec.parallel.item_similarity import ItemSimilarityRunner
    from mrec.parallel.wrmf import WRMFRunner
    from mrec.parallel.warp import WARPMFRunner

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('-n','--num_engines',dest='num_engines',type='int',default=0,help='number of IPython engines to use')
    parser.add_option('--input_format',dest='input_format',help='format of training dataset(s) tsv | csv | mm (matrixmarket) | fsm (fast_sparse_matrix)')
    parser.add_option('--train',dest='train',help='glob specifying path(s) to training dataset(s) IMPORTANT: must be in quotes if it includes the * wildcard')
    parser.add_option('--outdir',dest='outdir',help='directory for output files')
    parser.add_option('--overwrite',dest='overwrite',action='store_true',help='overwrite existing files in outdir')
    parser.add_option('--model',dest='model',default='slim',help='type of model to train: slim | knn | wrmf | warp | popularity (default: %default)')
    parser.add_option('--max_sims',dest='max_sims',type='int',default=100,help='max similar items to output for each training item (default: %default)')
    parser.add_option('--learner',dest='learner',default='sgd',help='underlying learner for SLIM learner: sgd | elasticnet | fs_sgd (default: %default)')
    parser.add_option('--l1_reg',dest='l1_reg',type='float',default=0.001,help='l1 regularization constant (default: %default)')
    parser.add_option('--l2_reg',dest='l2_reg',type='float',default=0.0001,help='l2 regularization constant (default: %default)')
    parser.add_option('--metric',dest='metric',default='cosine',help='metric for knn recommender: cosine | dot (default: %default)')
    parser.add_option('--num_factors',dest='num_factors',type='int',default=80,help='number of latent factors (default: %default)')
    parser.add_option('--alpha',dest='alpha',type='float',default=1.0,help='wrmf confidence constant (default: %default)')
    parser.add_option('--lbda',dest='lbda',type='float',default=0.015,help='wrmf regularization constant (default: %default)')
    parser.add_option('--als_iters',dest='als_iters',type='int',default=15,help='number of als iterations (default: %default)')
    parser.add_option('--gamma',dest='gamma',type='float',default=0.01,help='warp learning rate (default: %default)')
    parser.add_option('--C',dest='C',type='float',default=100.0,help='warp regularization constant (default: %default)')
    parser.add_option('--item_feature_format',dest='item_feature_format',help='format of item features tsv | csv | mm (matrixmarket) | npz (numpy arrays)')
    parser.add_option('--item_features',dest='item_features',help='path to sparse item features in tsv format (item_id,feature_id,val)')
    parser.add_option('--popularity_method',dest='popularity_method',default='count',help='how to compute popularity for baseline recommender: count | sum | avg | thresh (default: %default)')
    parser.add_option('--popularity_thresh',dest='popularity_thresh',type='float',default=0,help='ignore scores below this when computing popularity for baseline recommender (default: %default)')
    parser.add_option('--packer',dest='packer',default='json',help='packer for IPython.parallel (default: %default)')
    parser.add_option('--add_module_paths',dest='add_module_paths',help='optional comma-separated list of paths to append to pythonpath (useful if you need to import uninstalled modules to IPython engines on a cluster)')

    (opts,args) = parser.parse_args()
    if not opts.input_format or not opts.train or not opts.outdir or not opts.num_engines:
        parser.print_help()
        raise SystemExit

    opts.train = os.path.abspath(os.path.expanduser(opts.train))
    opts.outdir = os.path.abspath(os.path.expanduser(opts.outdir))

    trainfiles = glob.glob(opts.train)

    if opts.model == 'popularity':
        # special case, don't need to run in parallel
        subprocess.check_call(['mkdir','-p',opts.outdir])
        for trainfile in trainfiles:
            logging.info('processing {0}...'.format(trainfile))
            model = ItemPopularityRecommender(method=opts.popularity_method,thresh=opts.popularity_thresh)
            dataset = load_fast_sparse_matrix(opts.input_format,trainfile)
            model.fit(dataset)
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
    elif opts.model == 'wrmf':
        model = WRMFRecommender(d=opts.num_factors,alpha=opts.alpha,lbda=opts.lbda,num_iters=opts.als_iters)
    elif opts.model == 'warp':
        num_factors_per_engine = max(opts.num_factors/opts.num_engines,1)
        if opts.item_features:
            model = WARP2MFRecommender(d=num_factors_per_engine,gamma=opts.gamma,C=opts.C)
        else:
            model = WARPMFRecommender(d=num_factors_per_engine,gamma=opts.gamma,C=opts.C)
    else:
        parser.print_help()
        raise SystemExit('unknown model type: {0}'.format(opts.model))

    for trainfile in trainfiles:
        logging.info('processing {0}...'.format(trainfile))
        modelfile = get_modelfile(trainfile,opts.outdir)
        if opts.model == 'wrmf':
            runner = WRMFRunner()
            factorsdir = get_factorsdir(trainfile,opts.outdir)
            runner.run(view,model,opts.input_format,trainfile,opts.num_engines,factorsdir,modelfile)
        elif opts.model == 'warp':
            runner = WARPMFRunner()
            modelsdir = get_modelsdir(trainfile,opts.outdir)
            runner.run(view,model,opts.input_format,trainfile,opts.item_feature_format,opts.item_features,opts.num_engines,modelsdir,opts.overwrite,modelfile)
        else:
            runner = ItemSimilarityRunner()
            simsdir = get_simsdir(trainfile,opts.outdir)
            simsfile = get_simsfile(trainfile,opts.outdir)
            runner.run(view,model,opts.input_format,trainfile,opts.num_engines,simsdir,opts.overwrite,opts.max_sims,simsfile,modelfile)

if __name__ == '__main__':
    main()
