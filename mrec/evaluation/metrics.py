"""
Metrics to evaluate recommendations:
* with hit rate, following e.g. Karypis lab SLIM and FISM papers
* with prec@k and MRR
"""

import numpy as np
from scipy import stats
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
    - retrain(model,dataset) should retrain model
    - get_split() should return train_data,test_users,test_data
    - evaluation_func(model,users,test) should return a dict of metrics
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
    for name,vals in sorted(prefix2val.iteritems()):
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
            print '{0}{1:.4f} +/- {2:.4f}'.format(m.ljust(15),np.mean(vals),stats.sem(vals,ddof=0))

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
            'mrr':rr(recommended,known)}

def compute_hit_rate(recommended,known):
    if not known:
        return None
    return {'hit rate@10':hit_rate(recommended,known,10)}

# individual metrics

def prec(predicted,true,k,ignore_missing=False):
    """
    Compute precision@k.

    Parameters
    ==========
    predicted : array like
        Predicted items.
    true : array like
        True items.
    k : int
        Measure precision@k.
    ignore_missing : boolean (default: False)
        If True then measure precision only up to rank len(predicted)
        even if this is less than k, otherwise assume that the missing
        predictions were all incorrect

    Returns
    =======
    prec@k : float
        Precision at k.
    """
    if len(predicted) == 0:
        return 0
    correct = len(set(predicted[:k]).intersection(set(true)))
    num_predicted = k
    if len(predicted) < k and ignore_missing:
        num_predicted = len(predicted)
    return float(correct)/num_predicted

def hit_rate(predicted,true,k):
    """
    Compute hit rate i.e. recall@k assume a single test item.

    Parameters
    ==========
    predicted : array like
        Predicted items.
    true : array like
        Containing the single true test item.
    k : int
        Measure hit rate@k.

    Returns
    =======
    hitrate : int
        1 if true is amongst predicted, 0 if not.
    """
    if len(true) != 1:
        raise ValueError('can only evaluate hit rate for exactly 1 true item')
    return int(true[0] in predicted[:k])

def rr(predicted,true):
    """
    Compute Reciprocal Rank.

    Parameters
    ==========
    predicted : array like
        Predicted items.
    true : array like
        True items.

    Returns
    =======
    rr : float
        Reciprocal of rank at which first true item is found in predicted.

    Notes
    =====
    We'll under report this as our predictions are truncated.
    """
    for i,x in enumerate(predicted):
        if x in true:
            return 1.0/(i+1)
    return 0
