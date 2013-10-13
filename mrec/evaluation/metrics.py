"""
Metrics to evaluate recommendations:
* with hit rate, following e.g. Karypis lab SLIM and FISM papers
* with prec@k and MRR
"""

import numpy as np
from collections import defaultdict

# classes to access known items for each test user

class get_known_items_from_dict(object):

    def __init__(self,data):
        self.data = data

    def __call__(self,u):
        return self.data[u]

class get_known_items_from_csr_matrix(object):

    def __init__(self,data):
        self.data = data

    def __call__(self,u):
        return self.data[u].indices

class get_known_items_from_thresholded_csr_matrix(object):

    def __init__(self,data,min_value):
        self.data = data
        self.min_value = min_value

    def __call__(self,u):
        items = self.data[u].toarray().flatten()
        items[items<self.min_value] = 0
        return items.nonzero()

# methods to refit a model to a new training dataset

def retrain_recommender(model,dataset):
    model.fit(dataset)

# methods for metric computation itself

def run_evaluation(models,retrain,get_split,num_runs,evaluation_func):
    """
    This is the main entry point to run an evaluation.

    Supply functions to retrain model, to get a new split of data on
    each run, to get known items from the test set, and to compute the
    metrics you want:
      retrain(model,dataset) should retrain model
      get_split() should return train_data,test_users,test_data
      evaluation_func(model,users,test) should return a dict of metrics
    A number of suitable functions are already available in the module.
    """
    metrics = [defaultdict(list) for m in models]
    for _ in xrange(num_runs):
        train,users,test = get_split()
        for i,model in enumerate(models):
            retrain(model,train)
            run_metrics = evaluation_func(model,train,users,test)
            for m,val in run_metrics.iteritems():
                print m,val
                metrics[i][m].append(val)
    return metrics

def generate_metrics(get_known_items,compute_metrics):
    def evaluation_func(model,train,users,test):
        return evaluate(model,train,users,get_known_items(test),compute_metrics)
    return evaluation_func

def sort_metrics_by_name(names):
    # group by name and number in "@n"
    prefix2val = defaultdict(list)
    for name in names:
        parts = name.split('@')
        name = parts[0]
        if len(parts) > 1:
            val = int(parts[1])
            prefix2val[name].append(val)
        else:
            prefix2val[name] = []
    for name,vals in prefix2val.iteritems():
        prefix2val[name] = sorted(vals)
    ret = []
    for name,vals in prefix2val.iteritems():
        if vals:
            for val in vals:
                ret.append('{0}@{1}'.format(name,val))
        else:
            ret.append(name)
    return ret

def print_report(models,metrics):
    """
    Call this to print out the metrics returned by run_evaluation().
    """
    for model,results in zip(models,metrics):
        print model
        if hasattr(model,'similarity_matrix'):
            nnz = model.similarity_matrix.nnz
            num_items = model.similarity_matrix.shape[0]
            density = float(model.similarity_matrix.nnz)/num_items**2
            print 'similarity matrix nnz = {0} (density {1:.3f})'.format(nnz,density)
        for m in sort_metrics_by_name(results.keys()):
            vals = results[m]
            mean = np.mean(vals)
            std = np.std(vals)
            stderr = std/len(vals)**0.5
            print '{0}{1:.4f} +/- {2:.4f}'.format(m.ljust(15),mean,stderr)

def evaluate(model,train,users,get_known_items,compute_metrics):
    avg_metrics = defaultdict(float)
    count = 0
    for u in users:
        recommended = [r for r,_ in model.recommend_items(train,u,max_items=20)]
        metrics = compute_metrics(recommended,get_known_items(u))
        if metrics:
            for m,val in metrics.iteritems():
                avg_metrics[m] += val
            count += 1
    for m in avg_metrics:
        avg_metrics[m] /= float(count)
    return avg_metrics

# collections of metrics

def compute_main_metrics(recommended,known):
    if not known:
        return None
    return {'prec@5':prec(recommended,known,5),
            'prec@10':prec(recommended,known,10),
            'prec@15':prec(recommended,known,15),
            'prec@20':prec(recommended,known,20),
            'mrr':mrr(recommended,known)}

def compute_hit_rate(recommended,known):
    if not known:
        return None
    return {'hit rate@10':hit_rate(recommended,known,10)}

# individual metrics

def prec(predicted,true,k,fill=0):
    if not predicted:
        return 0
    correct = len(set(predicted[:k]).intersection(set(true)))
    ret = len(predicted[:k])
    # if fill==1 and the cutoff is larger than the number of docs retrieved,
    # then we assume nonrelevant docs fill in the rest (like trec_eval)
    return float(correct)/(ret+fill*(k-ret))

def hit_rate(predicted,true,k):
    assert(len(true)==1)
    return int(true[0] in predicted[:k])
    #return int(len(set(predicted[:k]).intersection(set(true)))>0)

def mrr(predicted,true):
    # TODO: NB we'll under report this as our predictions are truncated
    for i,x in enumerate(predicted):
        if x in true:
            return 1.0/(i+1)
    return 0
