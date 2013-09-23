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
      --binarize            binarize ratings (default: False)
      --normalize           scale training ratings to unit norm (default: False)
      --rating_thresh=RATING_THRESH
                            treat ratings below this as zero (default: 0)
      --test_size=TEST_SIZE
                            target number of test items for each user, if
                            test_size >= 1 treat as an absolute number, otherwise
                            treat as a fraction of the total items (default: 0.5)
      --discard_zeros       discard zero training ratings after thresholding
                            (default: False, not recommended, incompatible with
                            using training items to guarantee that recommendations
                            are novel)
      --sample_before_thresholding
                            choose test items before thresholding ratings
                            (default: False, not recommended, test items below
                            threshold will then be discarded)

TODO: binarization, normalization
TODO: test_size = 1 for hitrate metrics
TODO: other output formats?? other input formats???
