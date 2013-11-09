"""
CLiMF Collaborative Less-is-More Filtering, a variant of latent factor CF
which optimises a lower bound of the smoothed reciprocal rank of "relevant"
items in ranked recommendation lists.  The intention is to promote diversity
as well as accuracy in the recommendations.  The method assumes binary
relevance data, as for example in friendship or follow relationships.

CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
Yue Shi, Martha Larson, Alexandros Karatzoglou, Nuria Oliver, Linas Baltrunas, Alan Hanjalic
ACM RecSys 2012
"""

from math import exp, log
import random
import numpy as np

from mrec.mf.recommender import MatrixFactorizationRecommender


# TODO: cythonize most of this...


def g(x):
    """sigmoid function"""
    return 1/(1+exp(-x))

def dg(x):
    """derivative of sigmoid function"""
    return exp(x)/(1+exp(x))**2

class CLiMFRecommender(MatrixFactorizationRecommender):

    def __init__(self,d,lbda=0.01,gamma=0.01,max_iters=25):
        self.d = d
        self.lbda = lbda
        self.gamma = gamma
        self.max_iters = max_iters

    def fit(self,data):
        self.U = 0.01*np.random.random_sample((data.shape[0],self.d))
        self.V = 0.01*np.random.random_sample((data.shape[1],self.d))
        # TODO: create a validation set

        for iter in xrange(self.max_iters):
            print 'iteration {0}:'.format(iter+1)
            print 'objective = {0:.4f}'.format(self.objective(data))
            self.update(data)
            # TODO: compute MRR on validation set, terminate if appropriate

    def precompute_f(self,data,i):
        """
        precompute f[j] = <U[i],V[j]>

        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          i   : item of interest

        returns:
          dot products <U[i],V[j]> for all j in data[i]
        """
        items = data[i].indices
        f = dict((j,np.dot(self.U[i],self.V[j])) for j in items)
        return f

    def objective(self,data):
        """
        compute objective function F(U,V)

        params:
          data: scipy csr sparse matrix containing user->(item,count)
          U   : user factors
          V   : item factors
          lbda: regularization constant lambda
        returns:
          current value of F(U,V)
        """
        F = -0.5*self.lbda*(np.sum(self.U*self.U)+np.sum(self.V*self.V))
        for i in xrange(len(self.U)):
            f = self.precompute_f(data,i)
            for j in f:
                F += log(g(f[j]))
                for k in f:
                    F += log(1-g(f[k]-f[j]))
        return F

    def update(self,data):
        """
        update user/item factors using stochastic gradient ascent

        params:
          data : scipy csr sparse matrix containing user->(item,count)
          U    : user factors
          V    : item factors
          lbda : regularization constant lambda
          gamma: learning rate
        """
        for i in xrange(len(self.U)):
            dU = -self.lbda*self.U[i]
            f = self.precompute_f(data,i)
            for j in f:
                dV = g(-f[j])-self.lbda*self.V[j]
                for k in f:
                    dV += dg(f[j]-f[k])*(1/(1-g(f[k]-f[j]))-1/(1-g(f[j]-f[k])))*self.U[i]
                self.V[j] += self.gamma*dV
                dU += g(-f[j])*self.V[j]
                for k in f:
                    dU += (self.V[j]-self.V[k])*dg(f[k]-f[j])/(1-g(f[k]-f[j]))
            self.U[i] += self.gamma*dU

    def compute_mrr(self,data,test_users=None):
        """
        compute average Mean Reciprocal Rank of data according to factors

        params:
          data      : scipy csr sparse matrix containing user->(item,count)
          U         : user factors
          V         : item factors
          test_users: optional subset of users over which to compute MRR

        returns:
          the mean MRR over all users in data
        """
        mrr = []
        if test_users is None:
            test_users = range(len(self.U))
        for ix,i in enumerate(test_users):
            items = set(data[i].indices)
            if not items:
                continue
            predictions = np.sum(np.tile(self.U[i],(len(self.V),1))*self.V,axis=1)
            found = False
            for rank,item in enumerate(np.argsort(predictions)[::-1]):
                if item in items:
                    mrr.append(1.0/(rank+1))
                    found = True
                    break
            if not found:
                print 'fail, no relevant items predicted for test user {0}'.format(i+1)
                print 'known items: {0}'.format(items)
        assert(len(mrr) == len(test_users))
        return np.mean(mrr)

def main():
    import sys
    from mrec import load_sparse_matrix, save_recommender
    from mrec.mf.climf import CLiMFRecommender

    file_format = sys.argv[1]
    filepath = sys.argv[2]
    outfile = sys.argv[3]

    # load training set as scipy sparse matrix
    train = load_sparse_matrix(file_format,filepath)

    model = CLiMFRecommender(d=5)
    model.fit(train)

    save_recommender(model,outfile)

if __name__ == '__main__':
    import cProfile
    cProfile.run('main()')
