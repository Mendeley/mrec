.. _training:

======================
Training a recommender
======================

Here's the full list of basic options for ``mrec_train``::

    $ mrec_train
    Usage: mrec_train [options]

    Options:
      -h, --help            show this help message and exit
      -n NUM_ENGINES, --num_engines=NUM_ENGINES
                            number of IPython engines use
      --input_format=INPUT_FORMAT
                            format of training dataset(s): tsv | csv | mm (MatrixMarket) | fsm (fast_sparse_matrix)
      --train=TRAIN         glob specifying path(s) to training dataset(s)
                            IMPORTANT: must be in quotes if it includes the * wildcard
      --outdir=OUTDIR       directory for output files
      --overwrite           overwrite existing files in outdir (default: False)
      --recommender=RECOMMENDER
                            type of recommender: slim | knn | popularity
      --write_simsfile      write similar items to a tsv file as well as saving model (default: False)

The input file for training can hold the user-item matrix in a variety of formats.
You can specify more than one input file by passing a standard unix file glob
containing the * wildcard, for example specifying ``--train ml100-k/u*.binary`` will
train separate models for `ml-100k/u1.binary`, `ml-100k/u2.binary` and so on.  
This can be useful if you're doing cross-validation.

.. note::

    All input training files must have the same data format.  

A separate recommender will be trained for each input file, and saved to disk in the
specified output directory: if the input file is called ``ratings_1.tsv`` then the
recommender will be saved in the file ``ratings_1.tsv.model``, and so on.  The saved model
can be passed to the ``mrec_predict`` script, or used programmatically like
this::

    >>> model = load_recommender('ratings_1.tsv.model')
    >>> sims = model.get_similar_items(item_id)

You can supply additional options to ``mrec_train`` specifying parameter settings for the particular type of recommender you are training.
For a SLIM recommender you probably want to specify::

      --model=MODEL         underlying model to use (default: SGDRegressor)
      --l1_reg=L1_REG       l1 regularization constant (default: 0.01)
      --l2_reg=L2_REG       l2 regularization constant (default: 0.01)
      --max_sims=MAX_SIMS   max similar items to output for each training item
                            (default: 100)

For a k-nearest neighbour recommender you just need to supply::

      --max_sims=MAX_SIMS   max similar items to output for each training item
                            (default: 100)
      --metric=METRIC       distance metric for knn recommender: cosine | dot

In this case ``max_sims`` is simply passed to the constructor
of the ``KNNRecommender`` as the value of ``k``.

You can also train a baseline non-personalized recommender that just finds the most popular
items and recommends them to everybody. The options for this are::

      --max_sims=MAX_SIMS   max similar items to output for each training item
                            (default: 100)
      --popularity=POPULARITY
                            popularity measure to use: count | sum | avg
                            (default: count)
      --popularity_thresh=POPULARITY_THRESH
                            only consider ratings higher than this
                        
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


TODO: filename conventions


Parameter tuning for SLIM
-------------------------
Before training a SLIM recommender, you'll need to choose the regularization constants.
You can do this easily using the ``mrec_tune`` script, which computes similarity weights for some
sample items over a range of values for each constant, and picks the best combination based on some
simple parameters.  The 'best' regularization constants are those that give similarity weights
that are as sparse as possible, but not too sparse.  You run ``mrec_tune`` like this::

    $ mrec_tune -d splits/u.data.train.0 --input_format tsv --l1_min 0.001 --l1_max 1.0 --l2_min 0.0001 --l2_max 1 --max_sims 200 --min_sims 1 --max_sparse 0.3 --min_sims 1

This says that we want to find the best constants that result in no more than 200 similar items for each item,
provided no more than 30% of items have no similar items at all.  We'd like to explore combinations of regularization
constants where the l1 constant ranges from 0.001 to 1.0 and the l2 constant from 0.0001 to 1.0.
The script will run for a few seconds and then report the best settings::

    best parameter setting: {'l1_reg': 0.1, 'l2_reg': 0.001}
    mean # positive similarity weights per item = 96.0
    proportion of items with fewer than 1 positive similarity weights = 0.25
    mean # negative similarity weights per item = 43.4

