import math
import glob
import re
import os
import subprocess
from shutil import rmtree
import logging

from mrec import load_fast_sparse_matrix, save_recommender, load_recommender

class WARPRunner(object):

    def run(self,view,model,input_format,trainfile,num_engines,workdir,overwrite,modelfile):

        logging.info('creating models directory {0}...'.format(workdir))
        subprocess.check_call(['mkdir','-p',workdir])

        done = []
        if not overwrite:
            logging.info('checking for existing output models...')
            done.extend(self.find_done(workdir))
            if done:
                logging.info('found {0} output files'.format(len(done)))

        logging.info('creating tasks...')
        tasks = self.create_tasks(model,input_format,trainfile,workdir,num_engines,done)

        if tasks:
            logging.info('running in parallel across ipython engines...')
            async_job = view.map_async(process,tasks,retries=2)

            # wait for tasks to complete
            results = async_job.get()

            logging.info('checking output files...')
            done = self.find_done(workdir)
            remaining = len(tasks) - len(done)
        else:
            remaining = 0

        if remaining == 0:
            logging.info('SUCCESS: all tasks completed')
            logging.info('averaging {0} models...'.format(len(done)))
            for ix in sorted(done):
                # average two models at a time to limit memory usage
                partial_model = load_recommender(self.get_modelfile(ix,workdir))
                if ix == 0:
                    model = partial_model
                else:
                    model.U = (ix*model.U + partial_model.U)/float(ix+1)
                    model.V = (ix*model.V + partial_model.V)/float(ix+1)
            save_recommender(model,modelfile)
            logging.info('removing partial output files...')
            rmtree(workdir)
            logging.info('done')
        else:
            logging.error('FAILED: {0}/{1} tasks did not complete successfully'.format(remaining,len(tasks)))
            logging.error('try rerunning the command to retry the remaining tasks')

    def create_tasks(self,model,input_format,trainfile,outdir,num_engines,done):
        tasks = []
        for ix in xrange(num_engines):
            if ix not in done:
                outfile = self.get_modelfile(ix,outdir)
                tasks.append((model,input_format,trainfile,outfile,ix,num_engines))
        return tasks

    def find_done(self,outdir):
        success_files = glob.glob(os.path.join(outdir,'*.SUCCESS'))
        r = re.compile('.*?([0-9]+)\.model\.npz\.SUCCESS$')
        done = []
        for path in success_files:
            m = r.match(path)
            ix = int(m.group(1))
            done.append(ix)
        return done

    def get_modelfile(self,ix,workdir):
        return os.path.join(workdir,'{0}.model.npz'.format(ix))

def process(task):
    """
    Training task to run on an ipython engine.
    """

    # import modules required by engine
    import os
    import subprocess
    from mrec import load_sparse_matrix, save_recommender
    from mrec.mf.warp import ShuffleSampler

    model,input_format,trainfile,outfile,offset,step = task

    # TODO: configure this!!!
    positive_thresh = 1

    dataset = load_sparse_matrix(input_format,trainfile)
    # TODO: models don't seem to converge, investigate....
    #sampler = ShuffleSampler(dataset,positive_thresh,42,offset,step)
    sampler = ShuffleSampler(dataset,positive_thresh,42)

    model.fit(dataset,sampler)
    save_recommender(model,outfile)

    # record success
    cmd = ['touch','{0}.SUCCESS'.format(outfile)]
    subprocess.check_call(cmd)

    # return the offset for the samples that we've learned from
    return offset
