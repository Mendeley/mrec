import math
import glob
import re
import os
import subprocess
from shutil import rmtree
import logging

from mrec import load_sparse_matrix, save_recommender

class ItemSimilarityRunner(object):

    def run(self,view,model,input_format,trainfile,num_engines,simsdir,overwrite,max_sims,simsfile,modelfile):

        logging.info('finding number of items...')
        dataset = load_sparse_matrix(input_format,trainfile)
        num_users,num_items = dataset.shape
        del dataset
        logging.info('%d users and %d items', num_users, num_items)

        logging.info('creating sims directory {0}...'.format(simsdir))
        subprocess.check_call(['mkdir','-p',simsdir])

        done = []
        if not overwrite:
            logging.info('checking for existing output sims...')
            done.extend(self.find_done(simsdir))
            if done:
                logging.info('found {0} output files'.format(len(done)))

        logging.info('creating tasks...')
        tasks = self.create_tasks(model,input_format,trainfile,simsdir,num_items,num_engines,max_sims,done)

        if num_engines > 0:
            logging.info('running %d tasks in parallel across ipython'
                         ' engines...', len(tasks))
            async_job = view.map_async(process,tasks,retries=2)
            # wait for tasks to complete
            results = async_job.get()
        else:
            # Sequential run to make it easier for debugging
            logging.info('training similarity model sequentially')
            results = [process(task) for task in tasks]

        logging.info('checking output files...')
        done = self.find_done(simsdir)
        remaining = len(tasks) - len(done)
        if remaining == 0:
            logging.info('SUCCESS: all tasks completed')
            logging.info('concatenating {0} partial output files...'.format(len(done)))
            paths = [os.path.join(simsdir,'sims.{0}-{1}.tsv'.format(start,end)) for start,end in done]
            cmd = ['cat']+paths
            subprocess.check_call(cmd,stdout=open(simsfile,'w'))
            logging.info('removing partial output files...')
            rmtree(simsdir)
            logging.info('loading %d items in %s model from %s',
                         num_items, type(model).__name__, simsfile)
            model.load_similarity_matrix(simsfile,num_items)
            save_recommender(model,modelfile)
            logging.info('done')
        else:
            logging.error('FAILED: {0}/{1} tasks did not complete successfully'.format(remaining,len(tasks)))
            logging.error('try rerunning the command to retry the remaining tasks')

    def find_done(self,outdir):
        success_files = glob.glob(os.path.join(outdir,'*.SUCCESS'))
        r = re.compile('.*?([0-9]+)-([0-9]+)\.SUCCESS$')
        done = []
        for path in success_files:
            m = r.match(path)
            start = int(m.group(1))
            end = int(m.group(2))
            done.append((start,end))
        return done

    def create_tasks(self,model,input_format,trainfile,outdir,num_items,num_engines,max_similar_items,done):
        if num_engines == 0:
            # special marker for sequential run
            num_engines = 1
        items_per_engine = int(math.ceil(float(num_items)/num_engines))
        tasks = []
        for start in xrange(0,num_items,items_per_engine):
            end = min(num_items,start+items_per_engine)
            if (start,end) not in done:
                tasks.append((model,input_format,trainfile,outdir,start,end,max_similar_items))
        return tasks

def process(task):
    """
    Training task to run on an ipython engine.
    """

    # import modules required by engine
    import os
    import subprocess
    from mrec import load_fast_sparse_matrix

    model,input_format,trainfile,outdir,start,end,max_similar_items = task

    # initialise the model
    dataset = load_fast_sparse_matrix(input_format,trainfile)
    if hasattr(model,'similarity_matrix'):
        # clear out any existing similarity matrix to trigger recomputation of
        # the item-item similarities from the users' ratings.
        model.similarity_matrix = None

    # write sims directly to file as we compute them
    outfile = os.path.join(outdir,'sims.{0}-{1}.tsv'.format(start,end))
    out = open(outfile,'w')
    for j in xrange(start,end):
        w = model.get_similar_items(j,max_similar_items=max_similar_items,dataset=dataset)
        for k,v in w:
            print >>out,'{0}\t{1}\t{2}'.format(j+1,k+1,v)  # write as 1-indexed
    out.close()

    # record success
    cmd = ['touch',os.path.join(outdir,'{0}-{1}.SUCCESS'.format(start,end))]
    subprocess.check_call(cmd)

    # return the range that we've processed
    return start,end
