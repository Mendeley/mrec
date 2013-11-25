try:
    import cPickle as pickle
except ImportError:
    import pickle
import tempfile
import os
import numpy as np
from nose.tools import assert_less_equal
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_array_equal

from mrec.testing import get_random_coo_matrix

from mrec.base_recommender import BaseRecommender

class MyRecommender(BaseRecommender):
    def __init__(self):
        self.foo = np.ndarray(range(10))
        self.description = 'my recommender'
    def _create_archive(self):
        tmp = self.foo
        self.foo = None
        m = pickle.dumps(self)
        self.foo = tmp
        return {'model':m,'foo':self.foo}
    def _load_archive(self,archive):
        self.foo = archive['foo']

def save_load(r):
    f,path = tempfile.mkstemp(suffix='.npz')
    r.save(path)
    return BaseRecommender.load(path)

def check_read_description(r):
    f,path = tempfile.mkstemp(suffix='.npz')
    r.save(path)
    d = BaseRecommender.read_recommender_description(path)
    assert_equal(str(r),d)

def test_save_filepath_condition():
    r = BaseRecommender()
    invalid_filepath = 'no suffix'
    assert_raises(ValueError,r.save,invalid_filepath)

def test_save_load():
    r = save_load(BaseRecommender())
    assert_equal(type(r),BaseRecommender)
    r = MyRecommender()
    r2 = save_load(r)
    assert_equal(type(r2),type(r))
    assert_array_equal(r2.foo,r.foo)
    assert_equal(r2.description,r.description)

def test_read_recommender_description():
    check_read_description(BaseRecommender())
    check_read_description(MyRecommender())

def test_zero_known_item_scores():
    train = get_random_coo_matrix().tocsr()
    predictions = np.random.random_sample(train.shape)
    r = BaseRecommender()
    safe = r._zero_known_item_scores(predictions,train)
    num_users,num_items = predictions.shape
    for u in xrange(num_users):
        for i in xrange(num_items):
            if i in train[u].indices:
                assert_less_equal(safe[u,i],0)
            else:
                assert_equal(safe[u,i],predictions[u,i])
