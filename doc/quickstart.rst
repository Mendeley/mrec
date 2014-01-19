.. _quickstart:

=========================
Getting started with mrec
=========================

Install mrec
------------
You can most easily install `mrec` with pip::

    $ sudo pip install mrec

Installing from source
~~~~~~~~~~~~~~~~~~~~~~
Alternatively you can install `mrec` from source.  Installing `mrec` requires `numpy`, `scipy`, `scikit-learn`, `ipython`,
`cython` and `psutil`, and you'll also need `pyzmq` to run the utilities.
You can most easily install these using pip::

    $ sudo pip install numpy scipy scikit-learn cython ipython pyzmq psutil

You can then install `mrec` from source in the standard way::

    $ git clone https://github.com/Mendeley/mrec.git
    $ cd mrec
    $ sudo python setup.py install

This installs both the `mrec` library and the scripts described in the following sections.

.. note::

    You may want to specify where the scripts are installed::

    $ sudo python setup.py install --install-scripts /path/to/script/dir

Get some data
-------------
Let's start by grabbing a small dataset of movie ratings from the MovieLens project::

    $ wget http://www.grouplens.org/system/files/ml-100k.zip
    $ unzip ml-100k.zip

We'll work with the `u.data` file: this contains the ratings themselves in TSV format: user, item, rating, timestamp
(we'll be ignorning the timestamps)::

    $ head ml-100k/u.data
    196 242 3   881250949
    186 302 3   891717742
    22  377 1   878887116
    244 51  2   880606923
    166 346 1   886397596
    298 474 4   884182806
    115 265 2   881171488
    253 465 5   891628467
    305 451 3   886324817
    6   86  3   883603013

Get the data ready to use
-------------------------
To do useful work we need to split this dataset into `train` and `test` movies for each user.  The idea is that
we choose some items which the user rated and liked, and move them into the test set.  We then train our
recommender using only the remaining items for each user.  Once we've generated recommendations
we can evaluate them by seeing how many of the test items we've actually recommended.

Deciding which items a user liked involves taking some decisions about how to interpret rating scores (or
whatever other values you have in your input data - click counts, page views, and so on).  The Movielens
ratings run from 1 to 5 stars, so let's only put items in our test set if they have a score of 4 or 5.
We also have to decide how many of the items rated by each user we should put in the test set.  Selecting
too few test items means that we leave plenty of ratings for our recommender to learn from, but our evaluation
scores are likely to be low (as there are few "correct" test items that can be predicted) so may not give
us a very clear picture of whether one recommender is better than another.  Selecting too many test items means
that we don't leave enough training data for our recommender to learn anything.  For now let's put roughly
half of the movies that each user liked into the test set.

Run the ``mrec_prepare`` script to split the movies that users rated 4 or higher into roughly equal sized training and test
sets like this::

    $ mrec_prepare --dataset ml-100k/u.data --outdir splits --rating_thresh 4 --test_size 0.5 --binarize

This creates five different randomly chosen train/test splits::

    $ ls -lh splits/
    total 3.7M
    -rw-rw-r-- 1 mark mark 266K Sep 21 19:17 u.data.test.0
    -rw-rw-r-- 1 mark mark 266K Sep 21 19:17 u.data.test.1
    -rw-rw-r-- 1 mark mark 266K Sep 21 19:17 u.data.test.2
    -rw-rw-r-- 1 mark mark 266K Sep 21 19:17 u.data.test.3
    -rw-rw-r-- 1 mark mark 266K Sep 21 19:17 u.data.test.4
    -rw-rw-r-- 1 mark mark 474K Sep 21 19:17 u.data.train.0
    -rw-rw-r-- 1 mark mark 474K Sep 21 19:17 u.data.train.1
    -rw-rw-r-- 1 mark mark 474K Sep 21 19:17 u.data.train.2
    -rw-rw-r-- 1 mark mark 474K Sep 21 19:17 u.data.train.3
    -rw-rw-r-- 1 mark mark 474K Sep 21 19:17 u.data.train.4

If you look into any of these files you'll see that the ``--binarize`` option we gave to ``mrec_prepare``
has replaced the ratings with 0 or 1, depending whether or not the original rating met our chosen
threshold of 4.

Averaging evaluation results from each of these train/test splits should give us some reasonably trustworthy numbers.

.. note::

    You'll see that each test file is only half as big as the corresponding training file.
    That's because we only pick movies that the user liked to put into the test set.  The
    training files contain the other half of the movies that users liked, and *all* of
    the movies they didn't like. Even though our recommender won't try to learn a user's
    tastes from their low-rated
    movies, we need to leave them in the training data so that we don't end up
    recommending a movie that they've already seen.

For full details about using the ``mrec_prepare`` script see :ref:`Preparing training data <preparation>`.

Learn from the data
-------------------
Now you've prepared some data you can start training recommenders with the ``mrec_train`` script, but first
you'll need to start up some IPython engines to do the work::

    $ ipcluster start -n4 --daemonize

The ``-n4`` argument says that you want to start four engines.  In practice you'll want one engine for each core
you plan to use for processing.
If you don't specify ``-n``, ``ipcluster`` will start one engine for each core on your machine. That's fine, but
it's useful to know exactly how many engines are running.

Once the IPython engines are running you can kick off training a separate recommender for each train/test split
like this::

    $ mrec_train -n4 --input_format tsv --train "splits/u.data.train.*" --outdir models

This will run for a few seconds and you'll then find the trained models in the ``models`` directory::

    $ ls -lh models/
    total 17M
    -rw-rw-r-- 1 mark mark 1.4M Sep 21 19:48 u.data.train.0.model.npz
    -rw-rw-r-- 1 mark mark 2.1M Sep 21 19:48 u.data.train.0.sims.tsv
    -rw-rw-r-- 1 mark mark 1.4M Sep 21 19:48 u.data.train.1.model.npz
    -rw-rw-r-- 1 mark mark 2.1M Sep 21 19:48 u.data.train.1.sims.tsv
    -rw-rw-r-- 1 mark mark 1.4M Sep 21 19:48 u.data.train.2.model.npz
    -rw-rw-r-- 1 mark mark 2.1M Sep 21 19:48 u.data.train.2.sims.tsv
    -rw-rw-r-- 1 mark mark 1.4M Sep 21 19:48 u.data.train.3.model.npz
    -rw-rw-r-- 1 mark mark 2.1M Sep 21 19:48 u.data.train.3.sims.tsv
    -rw-rw-r-- 1 mark mark 1.4M Sep 21 19:48 u.data.train.4.model.npz
    -rw-rw-r-- 1 mark mark 2.1M Sep 21 19:48 u.data.train.4.sims.tsv

.. note::

    Alongside each model you'll see a file containing the item similarity matrix in TSV format.
    These can be useful if you want to inspect the similarity scores or use them outside of `mrec`,
    but they aren't essential and you can delete them if you want.

For more information about training recommenders with ``mrec_train`` see :ref:`Training a recommender <training>`.

Make some recommendations and evaluate them
-------------------------------------------
Now we have some trained models you can run the ``mrec_predict`` script to generate recommendations
and more importantly to evaluate them::

    $ mrec_predict --input_format tsv --test_input_format tsv --train "splits/u.data.train.*" --modeldir models --outdir recs

This will run for a few seconds printing out some progress information before showing the evaluation results::

    SLIM(SGDRegressor(alpha=0.101, epsilon=0.1, eta0=0.01, fit_intercept=False,
       l1_ratio=0.990099009901, learning_rate=invscaling,
       loss=squared_loss, n_iter=5, p=None, penalty=elasticnet,
       power_t=0.25, random_state=None, rho=None, shuffle=False, verbose=0,
       warm_start=False))
    mrr            0.6541 +/- 0.0023
    prec@5         0.4082 +/- 0.0016
    prec@10        0.3529 +/- 0.0010
    prec@15        0.3180 +/- 0.0009
    prec@20        0.2933 +/- 0.0008

This tells us that the recommender we trained was a SLIM model, based on scikit-learn's SGDRegressor.
The metrics shown are Mean Reciprocal Rank and Precision@k for a few values of k.  The precision values
are the easiest to understand: prec@5 of 0.4 means that on average two of the first five items recommended
to each user were found in the test set, i.e. they were movies that the user did really like.

You'll find the recommendations themselves in the `recs` directory::

    $ head recs/u.data.train.0.recs.tsv 
    237 100 0.22976178339
    237 194 0.215614718584
    237 174 0.205740941451
    237 318 0.199876443948
    237 357 0.190513438762
    237 195 0.188450807147
    237 480 0.16834165636
    237 197 0.167543389552
    237 181 0.166211624407
    237 134 0.164500008501

As you can see the first few recommendations from this run were for user 237, and our top recommendations
for him are movies 100, 194, 174, 318, 357.  If you're interested you can look these up in the u.item file
provided by MovieLens: they are `Fargo`, `The Sting`, `Raiders of the Lost Ark`, `Schindler's
List` and `One Flew Over the Cuckoo's Nest`.  The third column in the recommendations file is a predicted preference score.
It doesn't have a direct meaning, but higher is better.

For more details about making and evaluating recommendations with `mrec` see :ref:`Making and evaluating recommendations <evaluation>`.
