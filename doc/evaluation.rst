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
      -n NUM_ENGINES, --num_engines=NUM_ENGINES
                            number of IPython engines to use
      --input_format=INPUT_FORMAT
                            format of training dataset(s) tsv | csv | mm
                            (matrixmarket) | fsm (fast_sparse_matrix)
      --test_input_format=TEST_INPUT_FORMAT
                            format of test dataset(s) tsv | csv | mm
                            (matrixmarket) | npz (numpy binary)  (default: npz)
      --train=TRAIN         glob specifying path(s) to training dataset(s)
                            IMPORTANT: must be in quotes if it includes the *
                            wildcard
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


You can compute the same evaluation metrics for recommendations that have already been saved to disk, whether
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
                            
TODO: details of metrics
TODO: filename conventions, input file formats
