.. _training:

======================
Training a recommender
======================

Here are the basic options for ``mrec_train``::

    $ mrec_train
    Usage: mrec_train [options]

    Options:
      -h, --help            show this help message and exit
      -n NUM_ENGINES, --num_engines=NUM_ENGINES
                            number of IPython engines to use
      --input_format=INPUT_FORMAT
                            format of training dataset(s) tsv | csv | mm
                            (matrixmarket) | fsm (fast_sparse_matrix)
      --train=TRAIN         glob specifying path(s) to training dataset(s)
                            IMPORTANT: must be in quotes if it includes the *
                            wildcard
      --outdir=OUTDIR       directory for output files
      --overwrite           overwrite existing files in outdir
      --model=MODEL         type of model to train: slim | knn | wrmf | warp | popularity
                            (default: slim)
      --max_sims=MAX_SIMS   max similar items to output for each training item
                            (default: 100)

``mrec_train`` currently supports five types of recommender:

- `knn` learns a traditional k-nearest neighbours item similarity model
- `slim` specifies a SLIM model which learns item similarities by solving a regression problem
- `wrmf` fits a confidence-weighted matrix factorization model
- `warp` trains a model that optimizes a ranking loss, and can also learn from item features
- `popularity` is a trivial baseline that will make the same recommendations for all users, but which can be useful for evaluation.

The ``--train`` input file for training can hold the user-item matrix in a variety of formats.
You can specify more than one input file by passing a standard unix file glob
containing the * wildcard, for example specifying `--train ml100-k/u.train.*` will
train separate models for `ml-100k/u.train.0`, `ml-100k/u.train.1` and so on.  
This can be useful if you're doing cross-validation.

.. note::

    All input training files must have the same data format.  

A separate recommender will be trained for each input file, and saved to disk in the
specified output directory: if the input file is called `u.train.0` then the
recommender will be saved in the file `u.train.0.model.npz`, and so on.  See :ref:`filename_conventions-link` for more information.

The saved model
can be passed to the ``mrec_predict`` script as described in :ref:`evaluation`, or used programmatically like
this::

    >>> from mrec import load_sparse_matrix, load_recommender
    >>> train = load_sparse_matrix('tsv','u.train.0')
    >>> model = load_recommender('u.train.0.model.npz')
    >>> sims = model.get_similar_items(231)  # get items similar to 231
    >>> recs = model.recommend_items(train,101,max_items=30)  # recommend top 30 items for user 101

See :mod:`mrec.item_similarity.recommender` for more details.

You can supply additional options to ``mrec_train`` specifying parameter settings for the particular type of recommender you are training.
For a SLIM recommender you probably want to specify::

      --learner=LEARNER     underlying learner for SLIM learner: sgd | elasticnet
                            | fs_sgd (default: sgd)
      --l1_reg=L1_REG       l1 regularization constant (default: 0.1)
      --l2_reg=L2_REG       l2 regularization constant (default: 0.001)

For a k-nearest neighbour recommender you just need to supply::

      --metric=METRIC       metric for knn recommender: cosine | dot (default:
                        cosine)

In this case ``max_sims`` is simply passed to the constructor
of the ``KNNRecommender`` as the value of ``k``.

For the Weighted Regularized Matrix Factoriation (WRMF) recommender you can specify::

    --num_factors=NUM_FACTORS
                          number of latent factors (default: 80)
    --alpha=ALPHA         wrmf confidence constant (default: 1.0)
    --lbda=LBDA           wrmf regularization constant (default: 0.015)
    --als_iters=ALS_ITERS number of als iterations (default: 15)

The confidence constant determines the model's confidence in the rating/count associated
with an item using a simple linear formula::

    confidence = 1 + alpha * count

The regularization constant and number of learning iterations control over-fitting.

For the Weighted Approximately Ranked Pairwise (WARP) loss recommender the options are::

    --num_factors=NUM_FACTORS
                          number of latent factors (default: 80)
    --gamma=GAMMA         warp learning rate (default: 0.01)
    --C=C                 warp regularization constant (default: 100.0)
    --item_features=ITEM_FEATURES
                          path to sparse item features in tsv format
                          (item_id,feature_id,val)

The ``item_features`` option here is particularly interesting: if you supply a filepath
here then a hybrid recommender will be created, based on a model that learns
jointly from the item features in the file and from the ratings or preference scores in
the training user-item matrix. See :ref:`hybrid` for more details.

You can also train a baseline non-personalized recommender that just finds the most popular
items and recommends them to everybody. The options for this are::

    --popularity_method=POPULARITY_METHOD
                          how to compute popularity for baseline recommender:
                          count | sum | avg | thresh (default: count)
    --popularity_thresh=POPULARITY_THRESH
                          ignore scores below this when computing popularity for
                          baseline recommender (default: 0)
                        
The different measures mean let you base the popularity of an item on its total number of
ratings of any value, or its total above some threshold; or on the sum or mean of its ratings.

There are also a couple of options relating to the IPython.parallel framework::

    --packer=PACKER       packer for IPython.parallel (default: pickle)
    --add_module_paths=ADD_MODULE_PATHS
                          optional comma-separated list of paths to append to
                          pythonpath (useful if you need to import uninstalled
                          modules to IPython engines on a cluster)

The ``--add_module_paths`` option can be useful to specify the path to `mrec` itself
if you didn't install it at start up time on all the machines in your cluster.

Parameter tuning for SLIM
-------------------------
Before training a SLIM recommender, you'll need to choose the regularization constants.
You can do this easily using the ``mrec_tune`` script, which computes similarity weights for some
sample items over a range of values for each constant, and picks the best combination based on some
simple parameters.  The 'best' regularization constants are those that give similarity weights
that are as sparse as possible, but not too sparse.  You run ``mrec_tune`` like this::

    $ mrec_tune -d u.data.train.0 --input_format tsv \
        --l1_min 0.001 --l1_max 1.0 \
        --l2_min 0.0001 --l2_max 1 \
        --max_sims 200 --min_sims 1 --max_sparse 0.3

This says that we want to find the best constants that result in no more than 200 similar items for each item,
provided no more than 30% of items have no similar items at all.  We'd like to explore combinations of regularization
constants where the l1 constant ranges from 0.001 to 1.0 and the l2 constant from 0.0001 to 1.0.
The script will run for a few seconds and then report the best settings::

    best parameter setting: {'l1_reg': 0.1, 'l2_reg': 0.001}
    mean # positive similarity weights per item = 96.0
    proportion of items with fewer than 1 positive similarity weights = 0.25
    mean # negative similarity weights per item = 43.4

.. note::

    For this little dataset even the best constant values shown will mean that we won't learn
    any similar items for quite a large proportion of the training items.  This isn't
    usually a problem with production size datasets.
