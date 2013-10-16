import glob
import logging
import os
import subprocess
from shutil import rmtree
import math
import numpy as np

from mrec import load_sparse_matrix, save_recommender

def get_user_indices(data,u):
    # get (positive i.e. non-zero scored) items for user
    return data.X[u].nonzero()[1]

def get_item_indices(data,i):
    # get users for item
    return data.fast_get_col(i).nonzero()[0]

def get_factor_files(workdir,factor_type):
    # return partial factor files in sorted order so they can simply be stacked
    factor_files = glob.glob(os.path.join(workdir,'{0}.*.npy'.format(factor_type)))
    return sorted(factor_files,key=lambda x: int(x[:-4][x[:-4].rfind('.')+1:]))

def get_user_factor_files(workdir):
    return get_factor_files(workdir,'U')

def get_item_factor_files(workdir):
    return get_factor_files(workdir,'V')

def init_item_factors(model,data):
    num_users,num_items = data.shape
    return model.init_factors(num_items)

class WRMFRunner(object):

    def run(self,view,model,input_format,trainfile,num_engines,workdir,modelfile):
        logging.info('creating factors directory {0}'.format(workdir))
        subprocess.check_call(['mkdir','-p',workdir])

        logging.info('getting data size')
        data = load_sparse_matrix(input_format,trainfile)
        num_users,num_items = data.shape
        del data

        for it in xrange(model.num_iters):
            logging.info('iteration {0}'.format(it))
            tasks = self.create_tasks(num_users,num_engines,model,input_format,trainfile,workdir,'U',get_user_indices,get_item_factor_files,init_item_factors)
            self.run_tasks(view,tasks)
            tasks = self.create_tasks(num_items,num_engines,model,input_format,trainfile,workdir,'V',get_item_indices,get_user_factor_files,None)  # won't need to initialize user factors
            self.run_tasks(view,tasks)

        model.U = np.vstack([np.load(f) for f in get_user_factor_files(workdir)])
        model.V = np.vstack([np.load(f) for f in get_item_factor_files(workdir)])

        save_recommender(model,modelfile)

        logging.info('removing partial output files')
        rmtree(workdir)
        logging.info('done')

    def run_tasks(self,view,tasks):
        async_job = view.map_async(compute_factors,tasks,retries=2)
        # wait for tasks to complete
        result = async_job.get()

    def create_tasks(self,num_factors,num_engines,model,input_format,trainfile,workdir,factor_type,get_indices,get_fixed_factor_files,init_fixed_factors):
        factors_per_engine = int(math.ceil(float(num_factors)/num_engines))
        tasks = []
        for start in xrange(0,num_factors,factors_per_engine):
            end = min(num_factors,start+factors_per_engine)
            fixed_factor_files = get_fixed_factor_files(workdir)
            tasks.append((model,input_format,trainfile,factor_type,get_indices,init_fixed_factors,fixed_factor_files,start,end,workdir))
        return tasks

def compute_factors(task):
    """
    WRMF update method to run on an IPython engine.
    This reads from file and writes back to file,
    only filepaths and an empty model need to be passed.
    """

    # import modules needed on engine
    import os
    import numpy as np
    from mrec import load_fast_sparse_matrix

    model,input_format,trainfile,factor_type,get_indices,init_fixed_factors,fixed_factor_files,start,end,workdir = task

    data = load_fast_sparse_matrix(input_format,trainfile)

    if fixed_factor_files:
        H = np.vstack([np.load(f) for f in fixed_factor_files])
    else:
        H = init_fixed_factors(model,data)

    HH = H.T.dot(H)
    W = np.zeros(((end-start),model.d))
    for j in xrange(start,end):
        indices = get_indices(data,j)
        if indices.size:
            W[j-start,:] = model.update(indices,H,HH)

    np.save(os.path.join(workdir,'{0}.{1}.npy'.format(factor_type,start)),W)

    return start,end
