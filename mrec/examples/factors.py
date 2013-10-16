"""
Postprocess externally computed user/item factors so we can make
and evaluation recommendations with mrec scripts.
"""

def main():

    import os
    import logging
    import subprocess
    from optparse import OptionParser
    import numpy as np
    from scipy.io import mmread

    from mrec import save_recommender
    from mrec.mf.recommender import MatrixFactorizationRecommender
    from filename_conventions import get_modelfile

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('--factor_format',dest='factor_format',help='format of factor files tsv | mm (matrixmarket) | npy (numpy array)')
    parser.add_option('--user_factors',dest='user_factors',help='user factors filepath')
    parser.add_option('--item_factors',dest='item_factors',help='item factors filepath')
    parser.add_option('--train',dest='train',help='filepath to training data, just used to apply naming convention to output model saved here')
    parser.add_option('--outdir',dest='outdir',help='directory for output')
    parser.add_option('--description',dest='description',help='optional description of how factors were computed, will be saved with model so it can be output with evaluation results')

    (opts,args) = parser.parse_args()
    if not opts.factor_format or not opts.user_factors or not opts.item_factors \
            or not opts.outdir:
        parser.print_help()
        raise SystemExit

    model = MatrixFactorizationRecommender()

    logging.info('loading factors...')

    if opts.factor_format == 'npy':
        model.U = np.load(opts.user_factors)
        model.V = np.load(opts.item_factors)
    elif opts.factor_format == 'mm':
        model.U = mmread(opts.user_factors)
        model.V = mmread(opts.item_factors)
    elif opts.factor_format == 'tsv':
        model.U = np.loadtxt(opts.user_factors)
        model.V = np.loadtxt(opts.item_factors)
    else:
        raise ValueError('unknown factor format: {0}'.format(factor_format))

    if opts.description:
        model.description = opts.description

    logging.info('saving model...')

    logging.info('creating output directory {0}...'.format(opts.outdir))
    subprocess.check_call(['mkdir','-p',opts.outdir])

    modelfile = get_modelfile(opts.train,opts.outdir)
    save_recommender(model,modelfile)

    logging.info('done')

if __name__ == '__main__':
    main()
