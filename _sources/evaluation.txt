.. _evaluation:

=====================================
Making and evaluating recommendations
=====================================

Once you have a trained model, you can use the ``mrec_predict`` script to generate recommendations
and to evaluate them::

    $ mrec_predict
    Usage: mrec_predict [options]

    Options:
      -h, --help            show this help message and exit
      --mb_per_task=MB_PER_TASK
                            approximate memory limit per task in MB, so total
                            memory usage is num_engines * mb_per_task (default:
                            share all available RAM across engines)
      --input_format=INPUT_FORMAT
                            format of training dataset(s) tsv | csv | mm
                            (matrixmarket) | fsm (fast_sparse_matrix)
      --test_input_format=TEST_INPUT_FORMAT
                            format of test dataset(s) tsv | csv | mm
                            (matrixmarket) | npz (numpy binary)  (default: npz)
      --train=TRAIN         glob specifying path(s) to training dataset(s)
                            IMPORTANT: must be in quotes if it includes the *
                            wildcard
      --item_features=ITEM_FEATURES
                            path to sparse item features in tsv format
                            (item_id,feature_id,val)
      --modeldir=MODELDIR   directory containing trained models
      --outdir=OUTDIR       directory for output files
      --metrics=METRICS     which set of metrics to compute, main|hitrate
                            (default: main)
      --overwrite           overwrite existing files in outdir (default: False)
      --packer=PACKER       packer for IPython.parallel (default: json)
      --add_module_paths=ADD_MODULE_PATHS
                            optional comma-separated list of paths to append to
                            pythonpath (useful if you need to import uninstalled
                            modules to IPython engines on a cluster)

Even though you're making predictions with a recommender that has already been trained,
you need to specify the training file with the ``--train`` option so that the recommender
is able to exclude items that each user has already seen from their recommendations.
The corresponding test file used for evaluation is assumed to be in the same directory
as the training file, and with a related filepath following the convention described
in :ref:`filename_conventions-link`.

You only need to supply a filepath with the ``--item_features`` option if you used the
features during training.

You can choose one of two sets of metrics, the `main` metrics which include Precision@k
for various small values of `k` and Mean Reciprocal Rank, or `hitrate` which simply computes
the HitRate@10.  `hitrate` is only appropriate if your test set contains a single item for
each user; it measures how often the single test item appears in the top 10 recommendations, 
and is equivalent to Recall@10.

The recommendations themselves will be written to file in the ``--outdir``, in tsv format
`user`, `item`, `score`.  The `score` is not directly meaningful but higher is better for
when comparing two recommended items for the same user.

If your dataset is of any significant size, and particularly if your trained model is a
matrix factorization recommender, you may want to limit the amount of memory allocated by
each task to avoid OOM errors if you plan to do other work while ``mrec_predict`` is running.
You can do this with the ``--mb_per_task`` option: bear in
mind that the amount of memory specified with this option will be used concurrently on each
IPython engine.

Evaluating existing recommendations
-----------------------------------                            
For convenience the ``mrec_evaluate`` script lets you compute the same evaluation metrics for recommendations that have already been saved to disk, whether
with ``mrec_predict`` or some other external program::

    $ mrec_evaluate
    Usage: mrec_evaluate [options]

    Options:
      -h, --help            show this help message and exit
      --input_format=INPUT_FORMAT
                            format of training dataset(s) tsv | csv | mm
                            (matrixmarket) | fsm (fast_sparse_matrix)
      --test_input_format=TEST_INPUT_FORMAT
                            format of test dataset(s) tsv | csv | mm
                            (matrixmarket) | npz (numpy binary)  (default: npz)
      --train=TRAIN         glob specifying path(s) to training dataset(s)
                            IMPORTANT: must be in quotes if it includes the *
                            wildcard
      --recsdir=RECSDIR     directory containing tsv files of precomputed
                            recommendations
      --metrics=METRICS     which set of metrics to compute, main|hitrate
                            (default: main)
      --description=DESCRIPTION
                            description of model which generated the
                            recommendation
                            
