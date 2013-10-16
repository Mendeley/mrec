import numpy as np
import random
import math
from bisect import bisect_left
from collections import defaultdict

from recommender import MatrixFactorizationRecommender

class Sampler(object):
    """
    Pair sampler for WARP loss estimation.

    Parameters
    ==========
    data : scipy.sparse.csr_matrix
        The training ratings to sample.
    positive_thresh : int
        Positive items must have rating at least equal to this.
    """

    def __init__(self,data,positive_thresh):
        self.data = data
        self.num_users,self.num_items = data.shape
        self.positive_thresh = positive_thresh
        # say we put items for each user in a hashmap
        self.user2item = defaultdict(dict)
        for u in xrange(self.num_users):
            begin,end = self.data.indptr[u],self.data.indptr[u+1]
            self.user2item[u].update(zip(self.data.indices[begin:end],self.data.data[begin:end]))

    def sample_positive_example(self):
        """
        All Samplers must implement this.
        """
        pass

    def sample_worse_example(self,u,ix):
        #begin,end = self.data.indptr[u],self.data.indptr[u+1]
        #if not (end-begin > 1 and end-begin < self.num_items):
        #    # won't be able to find a worse example
        #    return None
        num_items = len(self.user2item[u])
        if num_items == 0 or num_items == self.num_items:
            print 'WARNING: no worse examples for user {0} because num_items = {1}'.format(u,num_items)
            return None
        while True:
            # sample item uniformly with replacement
            j = random.randint(0,self.num_items-1)
            ## find where j would be in the training data if it is in u's items
            #jx = begin + bisect_left(self.data.indices[begin:end],j)
            ## check that it's not there at all, or is there but has a worse score
            #if jx == end or j != self.data.indices[jx] or self.data.data[jx] < self.data.data[ix]:
            #    return j
            r = self.user2item[u].get(j,None)
            if r is None or r < self.data.data[ix]:
                return j

    def sample_worse_examples(self,u,ix):
        """
        Return a bunch of worse examples
        - what's the most we could do?
        return all of the u's lower rated items
        and all of his unrated items
        - we're then going to do a dot product
        so there may not be much cost in returning a lot
        for simplicity perhaps we could stochastically
        return either one category or the other
        """
        # TODO: is it better to return all the non-worse items???
        # then we can just sample at random from all items and
        # try again if we get a non-worse one, adding each one
        # we find and use into the exclude set????
        begin,end = self.data.indptr[u],self.data.indptr[u+1]
        if random.uniform(0,1) < float(end-begin)/self.num_items:
            # find lower rated items
            ratings = self.data.data[begin:end]
            return self.data.indices[ratings < self.data.data[ix]]
        else:
            # return unrated items
            all_items = set(range(self.num_items))
            return list(all_items - set(self.data.indices[begin:end]))

class UniformUserSampler(Sampler):
    """
    Pair sampler for WARP loss estimation. First selects a user
    uniformly at random with replacement, then a qualifying item pair.

    Parameters
    ==========
    data : scipy.sparse.csr_matrix
        The training ratings to sample.
    positive_thresh : int
        Positive items must have rating at least equal to this.
    """

    def sample_positive_example(self):
        while True:
            # pick a random user
            u = random.randint(0,self.num_users-1)
            if self.data.indptr[u+1] == self.data.indptr[u]:
                continue  # this user has no items in the training set
            # pick one of their items
            ix = random.randint(self.data.indptr[u],self.data.indptr[u+1]-1)
            i = self.data.indices[ix]
            r = self.data.data[ix]
            if r >= self.positive_thresh:
                break
        return u,ix,i,r

class EmpiricalSampler(Sampler):
    """
    Pair sampler for WARP loss estimation. First selects a <user,positive item> pair
    uniformly at random with replacement, then finds a qualifying worse item.

    Parameters
    ==========
    data : scipy.sparse.csr_matrix
        The training ratings to sample.
    positive_thresh : int
        Positive items must have rating at least equal to this.
    """

    def sample_positive_example(self):
        # pick a random training data point
        while True:
            ix = random.randint(0,self.data.nnz-1)
            r = self.data.data[ix]
            if r >= self.positive_thresh:
                break
        i = self.data.indices[ix]
        # find the user it belongs to
        for u in xrange(self.num_users):
            if ix < self.data.indptr[u+1]:
                break
        return u,ix,i,r

class ShuffleSampler(Sampler):
    """
    Pair sampler for WARP loss estimation. First select a <user,positive item>
    pair at random without replacement, then finds a qualifying worse item.

    The random_seed, offset and step parameters allow you to create a pool
    of ShuffleSamplers guaranteed to return disjoint samples.

    Parameters
    ==========
    data : scipy.sparse.csr_matrix
        The training ratings to sample.
    positive_thresh : int
        Positive items must have rating at least equal to this.
    random_seed : int (default: None)
        Optional random seed to guarantee identical sampling order for
        multiple samplers, useful together with offset and step options.
        If None then random.seed() is not called at all.
    offset : int (default: 0)
        Start sampling from this <user,positive item> pair after shuffling.
    step : int (default : 1)
        Interval between successive samples after shuffling.
    """

    def __init__(self,data,positive_thresh,random_seed=None,offset=0,step=1):
        Sampler.__init__(self,data,positive_thresh)
        if random_seed is not None:
            random.seed(random_seed)
        self.order = np.arange(self.data.nnz)
        self.order = self.order[self.data.data>=positive_thresh]
        random.shuffle(self.order)
        self.j = offset
        self.offset = offset
        self.step = step

    def sample_positive_example(self):
        if self.j >= len(self.order):
            self.j = self.offset
        ix = self.order[self.j]
        r = self.data.data[ix]
        i = self.data.indices[ix]
        # find the user it belongs to
        for u in xrange(self.num_users):
            if ix < self.data.indptr[u+1]:
                break
        self.j += self.step
        return u,ix,i,r

class WARPMFRecommender(MatrixFactorizationRecommender):
    """
    Learn matrix factorization optimizing the WARP loss.

    Parameters
    ==========
    d : int
        Dimensionality of factors.
    gamma : float
        Learning rate.
    C : float
        Regularization constant.
    max_iter : int
        Terminate after this number of iterations even if validation loss is still decreasing.
    validation_iters : int
        Check validation loss once each validation_iters iterations, terminate if it
        has increased.
    """

    def __init__(self,d,gamma,C,max_iter,validation_iters):
        self.d = d  # embedding dimension
        self.gamma = gamma # learning rate
        self.C = C  # regularization constant
        self.max_iter = max_iter
        self.validation_iters = validation_iters # check validation error after this many iterations
        #self.minibatch_size = 1

    def __str__(self):
        return 'WARPMF(d={0},gamma={1},C={2},max_iter={3},validation_iters={4})'.format(self.d,self.gamma,self.C,self.max_iter,self.validation_iters)

    def fit(self,train,sampler=None):
        """
        Learn factors from training set.

        Parameters
        ==========
        train : scipy.sparse.csr_matrix
            User-item matrix.
        sampler : mrec.mf.warp.Sampler (default: None)
            Sampler to provide rating pairs, if None a ShuffleSampler(train,1)
            will be used.
        """
        self._init(train)
        # precompute WARP loss for each possible rank
        self.warp_loss = np.ones(self.num_items)
        assert(self.num_items>1)
        for i in xrange(1,self.num_items):
            self.warp_loss[i] = self.warp_loss[i-1]+1.0/(i+1)
        if sampler is None:
            self.sampler = ShuffleSampler(train,1)
        else:
            self.sampler = sampler
        self.U = self.d**-0.5*np.random.random_sample((self.num_users,self.d))
        self.V = self.d**-0.5*np.random.random_sample((self.num_items,self.d))
        num_validation_samples = 100 #self.data.nnz/100
        # TODO: if we use the shuffle sampler maybe we don't
        # update the validation samples again during the epoch
        # which is lots and lots of iters...????
        #validation_sampler = UniformUserSampler(train,1)
        # TODO: nah doesn't seem to be that, still odd that we don't see
        # the validation loss moving at all at the learning rates that
        # give best evaluation results... maybe we need another thing
        # (very different C???) to avoid over fitting??
        # anyway need to try this on a bigger dataset now!
        self.validation_samples = [self.sampler.sample_positive_example() for _ in xrange(num_validation_samples)]
        prev_err = None
        gamma = self.gamma
        for it in xrange(self.max_iter):
            #print 'iteration',it
            if it % self.validation_iters == 0:
                err = self.validation_loss()
                print '{0}: validation loss = {1}'.format(it,err)
                if prev_err is not None and err > prev_err * 1.001:
                    print 'validation error got worse, terminating'
                    break
                prev_err = err

            u,ix,i,r = self.sampler.sample_positive_example()
            L,j = self.estimate_warp_loss(u,ix,i,r)
            if L is None:
                continue
            #print 'making gradient update'
            # make a gradient update
            dU = L*(self.V[i]-self.V[j])
            dV_i = L*self.U[u]
            dV_j = -L*self.U[u]
            self.U[u] += gamma*dU
            self.V[i] += gamma*dV_i
            self.V[j] += gamma*dV_j
            # regularize
            self.regularize(self.U,u)
            self.regularize(self.V,i)
            self.regularize(self.V,j)
            """
            # make a gradient update
            U_update,V_update = self.minibatch()
            for u,dU in U_update:
                self.U[u] += gamma*dU
            for v,dV in V_update:
                self.V[v] += gamma*dV

            # regularize
            for u,_ in U_update:
                self.regularize(self.U,u)
            for v,_ in V_update:
                self.regularize(self.V,v)
            """

            # update learning rate
            # TODO: only do this once per epoch otherwise rate falls to zero
            # too quickly with sgd or minibatch
            #gamma *= 0.99999

    def minibatch(self):
        U_update = []
        V_update = []
        for it in xrange(self.minibatch_size):
            u,ix,i,r = self.sampler.sample_positive_example()
            L,j = self.estimate_warp_loss(u,ix,i,r)
            if L is None:
                continue
            # make a gradient update
            dU = L*(self.V[i]-self.V[j])
            dV_i = L*self.U[u]
            dV_j = -L*self.U[u]
            U_update.append((u,dU))
            V_update.append((i,dV_i))
            V_update.append((j,dV_j))
        return U_update,V_update

    def estimate_warp_loss(self,u,ix,i,r):
        """
        N = 0
        while N < self.num_items:
            N += 1
            # find j!=i s.t. data[u,j] < data[u,i]
            j = self.sampler.sample_worse_example(u,ix)
            if j == None:
                # couldn't find a worse example for this user
                return None,None
            if np.dot(self.U[u],self.V[j]) > np.dot(self.U[u],self.V[i]) - 1:
                # compute WARP loss
                estimated_rank = (self.num_items-1)/N
                return self.warp_loss[estimated_rank],j
            #else:
            #    print 'not worse than WARP thresh for',u,i,j,np.dot(self.U[u],self.V[i]),np.dot(self.U[u],self.V[j])
        return None,None
        """
        worse = self.sampler.sample_worse_examples(u,ix)
        scores = np.dot(self.V[worse],self.U[u])
        r = np.dot(self.V[i],self.U[u])
        for N,j in enumerate(worse):
            #print scores[N],r-1
            if scores[N] > r - 1:
                # compute WARP loss
                estimated_rank = (self.num_items-1)/(N+1)
                return self.warp_loss[estimated_rank],j
        return None,None

    def validation_loss(self):
        loss = 0
        for u,ix,i,r in self.validation_samples:
            L,_ = self.estimate_warp_loss(u,ix,i,r)
            if L is not None:
                loss += L
        return loss

    def regularize(self,F,ix):
        """
        project back to enforce regularization constraint
        on ix-th row of matrix F
        """
        p = np.dot(F[ix].T,F[ix])**0.5/self.C
        if p > 1:
            F[ix] /= p  # ensure ||F[ix]|| <= C

def main():
    import sys
    from mrec import load_sparse_matrix
    from mrec.sparse import fast_sparse_matrix

    # load training set as scipy sparse matrix
    file_format = sys.argv[1]
    filepath = sys.argv[2]
    train = load_sparse_matrix(file_format,filepath)

    model = WARPMFRecommender(d=100,gamma=0.01,C=100,max_iter=100000,validation_iters=500)  # these values work for ml-100k
    sampler = ShuffleSampler(train,1)
    model.fit(train,sampler)

    def output(i,j,val):
        # convert back to 1-indexed
        print '{0}\t{1}\t{2:.3f}'.format(i+1,j+1,val)

    print 'making some recommendations...'
    for u in xrange(20):
        recs = model.recommend_items(train,u)
        for i,score in recs:
            output(u,i,score)

    print 'making batch recommendations...'
    recs = model.batch_recommend_items(train)
    for u in xrange(20):
        for i,score in recs[u]:
            output(u,i,score)

    print 'making range recommendations...'
    for start,end in [(0,2),(2,3)]:
        recs = model.range_recommend_items(train,start,end)
        for u in xrange(start,end):
            for i,score in recs[u-start]:
                output(u,i,score)

if __name__ == '__main__':
    main()
