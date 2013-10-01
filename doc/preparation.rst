.. _preparation:

=======================
Preparing training data
=======================

Run the ``mrec_prepare`` script to create train/test splits from a ratings dataset in TSV format.
Each line should contain: user, item, score.  Any further fields in each line will be ignored::

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

Filename conventions
--------------------
``mrec_prepare`` and the other `mrec` scripts rely on a set of filename conventions defined
in the :ref:`filename_conventions <filename_conventions-label>` module.
