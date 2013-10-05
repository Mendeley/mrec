class Processor(object):

    def __init__(self,splitter,parser,min_items_per_user,preprocess=None):
        self.splitter = splitter
        self.parser = parser
        self.min_items_per_user = min_items_per_user
        self.preprocess = preprocess

    def output(self,user,vals,outfile):
        for v,c in vals:
            print >>outfile,'{0}\t{1}\t{2}'.format(user,v,c)

    def handle(self,user,vals):
        if len(vals) >= self.min_items_per_user:
            if self.preprocess is not None:
                vals = self.preprocess(vals)
            train,test = self.splitter.handle(vals)
            self.output(user,train,self.train_out)
            self.output(user,test,self.test_out)
        else:
            self.too_few_items += 1

    def create_split(self,infile,train_out,test_out):
        self.train_out = train_out
        self.test_out = test_out
        self.too_few_items = 0
        last_user = None
        vals = []
        for line in infile:
            user,val = self.parser.parse(line)
            if user != last_user:
                if last_user is not None:
                    self.handle(last_user,vals)
                last_user = user
                vals = []
            vals.append(val)
        self.handle(last_user,vals)

    def get_too_few_items(self):
        return self.too_few_items

def main():
    import os
    import logging
    import subprocess
    from optparse import OptionParser

    from mrec.evaluation.preprocessing import TSVParser, SplitCreator
    from filename_conventions import get_sortedfile, get_splitfile

    logging.basicConfig(level=logging.INFO,format='[%(asctime)s] %(levelname)s: %(message)s')

    parser = OptionParser()
    parser.add_option('--dataset',dest='dataset',help='path to input dataset in tsv format')
    parser.add_option('--delimiter',dest='delimiter',default='\t',help='input delimiter (default: tab)')
    parser.add_option('--outdir',dest='outdir',help='directory for output files')
    parser.add_option('--num_splits',dest='num_splits',type='int',default=5,help='number of train/test splits to create (default: %default)')
    parser.add_option('--min_items_per_user',dest='min_items_per_user',type='int',default=10,help='skip users with less than this number of ratings (default: %default)')
    parser.add_option('--binarize',dest='binarize',action='store_true',default=False,help='binarize ratings')
    parser.add_option('--normalize',dest='normalize',action='store_true',help='scale training ratings to unit norm')
    parser.add_option('--rating_thresh',dest='rating_thresh',type='float',default=0,help='treat ratings below this as zero (default: %default)')
    parser.add_option('--test_size',dest='test_size',type='float',default=0.5,help='target number of test items for each user, if test_size >= 1 treat as an absolute number, otherwise treat as a fraction of the total items (default: %default)')
    parser.add_option('--discard_zeros',dest='discard_zeros',action='store_true',help='discard zero training ratings after thresholding (not recommended, incompatible with using training items to guarantee that recommendations are novel)')
    parser.add_option('--sample_before_thresholding',dest='sample_before_thresholding',action='store_true',help='choose test items before thresholding ratings (not recommended, test items below threshold will then be discarded)')

    (opts,args) = parser.parse_args()
    if not opts.dataset or not opts.outdir:
        parser.print_help()
        raise SystemExit

    opts.dataset = os.path.abspath(opts.dataset)
    opts.outdir = os.path.abspath(opts.outdir)

    logging.info('sorting input data...')
    infile = get_sortedfile(opts.dataset,opts.outdir)
    subprocess.check_call(['mkdir','-p',opts.outdir])
    subprocess.check_call(['sort','-k1','-n',opts.dataset],stdout=open(infile,'w'))

    parser = TSVParser(thresh=opts.rating_thresh,binarize=opts.binarize,delimiter=opts.delimiter)
    splitter = SplitCreator(test_size=opts.test_size,normalize=opts.normalize,discard_zeros=opts.discard_zeros,
                            sample_before_thresholding=opts.sample_before_thresholding)
    processor = Processor(splitter,parser,opts.min_items_per_user)

    for i in xrange(opts.num_splits):
        trainfile = get_splitfile(opts.dataset,opts.outdir,'train',i)
        testfile = get_splitfile(opts.dataset,opts.outdir,'test',i)

        logging.info('creating split {0}: {1} {2}'.format(i,trainfile,testfile))
        processor.create_split(open(infile),open(trainfile,'w'),open(testfile,'w'))

        too_few_items = processor.get_too_few_items()
        if (too_few_items):
            logging.info('skipped {0} users with less than {1} ratings'.format(too_few_items,opts.min_items_per_user))

    logging.info('cleaning up...')
    subprocess.check_call(['rm',infile])
    logging.info('done')

if __name__ == '__main__':
    main()

