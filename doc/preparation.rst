.. _preparation:

=======================
Preparing training data
=======================

Run the ``mrec_prepare`` script to create train/test splits from a ratings dataset in TSV format.
Each line should contain: `user`, `item`, `score`. `user` and `item` should be integer IDs starting from 1, and `score` is a rating or some other value describing how much the user likes or has interacted with the item.  Any further fields in each line will be ignored::

    $ mrec_prepare
    Usage: mrec_prepare [options]

    Options:
      -h, --help            show this help message and exit
      --dataset=DATASET     path to input dataset in tsv format
      --outdir=OUTDIR       directory for output files
      --num_splits=NUM_SPLITS
                            number of train/test splits to create (default: 5)
      --min_items_per_user=MIN_ITEMS_PER_USER
                            skip users with less than this number of ratings
                            (default: 10)
      --binarize            binarize ratings
      --normalize           scale training ratings to unit norm
      --rating_thresh=RATING_THRESH
                            treat ratings below this as zero (default: 0)
      --test_size=TEST_SIZE
                            target number of test items for each user, if
                            test_size >= 1 treat as an absolute number, otherwise
                            treat as a fraction of the total items (default: 0.5)
      --discard_zeros       discard zero training ratings after thresholding (not
                            recommended, incompatible with using training items to
                            guarantee that recommendations are novel)
      --sample_before_thresholding
                            choose test items before thresholding ratings (not
                            recommended, test items below threshold will then be
                            discarded)


The options are designed to support various common training and evaluation scenarios.

Rating preprocessing options
----------------------------
If you plan to train a SLIM recommender then you most likely need to ``--binarize`` or
``--normalize`` ratings to get good results.  You may also want to set
a global ``--rating_thresh`` so that an item to which a user has given a low ratings is
not considered as 'liked' by that user; ratings below the specified threshold are set
to zero.

Split options
-------------
To evaluate a recommender you need to hide some of the items that were liked by each user
by removing them to a test set.  Then you can generate recommendations based on the remaining
training ratings, and see how many of the test items were successfully recommended.
It will be hard to get meaningful results for users with very few rated items so these
can simply be skipped by setting ``--min_items_per_user``.  Usually you'll want to create
several train/test splits at random using the ``--num_splits`` option and then average evaluation
results across them, so that the results aren't biased by the particular way in which any
one split happens to be chosen.

You can choose how many items to move into the test set for
each user with the ``--test_size`` option.  A typical choice for this is 0.5, which puts
half of each users liked items into the test set, but you can vary this if you need to compare
with previous results that used a different split.  You can also specify an absolute number
of ratings by setting ``--test_size`` to an integer of 1 or more.  This is also useful if you
plan to measure :ref:`Hit Rate <evaluation>` in which case you should specify ``--test_size 1``.

.. _filename_conventions-link:

Filename conventions
--------------------
``mrec_prepare`` and the other `mrec` scripts use a set of filename conventions defined
in the :mod:`mrec.examples.filename_conventions` module:

- input dataset: `ratings.tsv`
- training files created by ``mrec_prepare``: `ratings.tsv.train.0`, `ratings.tsv.train.1`, ...
- test files created by ``mrec_prepare``: `ratings.tsv.test.0`, `ratings.tsv.test.1`, ...
- models created by ``mrec_train``: `ratings.tsv.train.0.model.npz`, `ratings.tsv.train.1.model.npz`, ...
- recommendations created by ``mrec_predict``: `ratings.tsv.train.0.recs.tsv`, `ratings.tsv.train.1.recs.tsv`, ...
