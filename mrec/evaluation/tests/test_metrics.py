from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_raises

from mrec.evaluation import metrics

def test_sort_metrics_by_name():
    names = ['recall@10','z-score','auc','recall@5']
    expected = ['auc','recall@5','recall@10','z-score']
    assert_equal(expected,metrics.sort_metrics_by_name(names))

def test_prec():
    true = [2,8,6,4]
    predicted = [6,5,8,7]
    expected = [1,0.5,2./3.,0.5]
    for k in xrange(1,5):
        assert_equal(metrics.prec([],true,k),0)
        assert_equal(metrics.prec(true,true,k),1)
        assert_equal(metrics.prec(predicted,true,k),expected[k-1])
    assert_equal(metrics.prec(true,true,5),0.8)
    assert_equal(metrics.prec(true,true,5,ignore_missing=True),1)
    assert_equal(metrics.prec(predicted,true,5),0.4)
    assert_equal(metrics.prec(predicted,true,5,ignore_missing=True),expected[3])

def test_hit_rate():
    predicted = [6,5,8,7]
    for true in [[],[2,8]]:
        for k in xrange(1,5):
            with assert_raises(ValueError):
                metrics.hit_rate(predicted,true,k)
    true = [5]
    expected = [0,1,1,1]
    for k in xrange(1,5):
        assert_equal(metrics.hit_rate(predicted,true,k),expected[k-1])

def test_rr():
    true = [2,8,6,4]
    predicted = [5,7,6,8]
    expected = [0,0,1./3.,1./3.]
    for k in xrange(1,5):
        assert_equal(metrics.rr(predicted[:k],true),expected[k-1])
