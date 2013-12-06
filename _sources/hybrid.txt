.. _hybrid:

===================================
Learning jointly from item features
===================================

In real world settings it's common to have features describing each item as well as ratings or
other counts expressing users historical interactions with items. As we expect that users might
like items with similar features to those that they have liked in the past, it should be useful
for a recommender to take item features into account. One way of doing this is to extend the
matrix factorization approach, which represents each user and item with a low-dimensional vector,
by learning to represent each feature in the same low-dimensional space. ``mrec`` includes
an implementation of a :class:`joint model <mrec.mf.model.warp2.WARP2>` of this kind which optimizes
the WARP ranking loss [1]_. This model learns an embedding matrix which maps the item features into
the low-dimensional space. To predict the rating or preference score for an unseen item, it
computes the dot product of the user factor and item factor in the usual way for a matrix
factorization recommender, but then adds the dot product of the user factor and the
low-dimensional mapping of its feature vector.

As an example we'll look again at the small movie ratings dataset that we previously worked with
in :ref:`quickstart`, but this time we'll add some features based on movie plot descriptions
from IMDb. To create the features, first download the plot.list.gz file from one of the `official IMDb ftp sites <http://www.imdb.com/interfaces#plain>`_. This contains plot summaries for most of the movies
in the MovieLens datasets. Once you've unzipped this file you can use the ``extract_movie_features``
script in the bin directory of the ``mrec`` source tree to create features and save them to file::
    
    $ cd mrec
    $ ./bin/extract_movie_features plot.list ml-100k/u.item 100k.features.npz

.. note::

    The ``extract_movie_features`` script isn't installed automatically with ``mrec`` so
    you'll need to `grab the source code <https://github.com/mendeley/mrec>`_ if you don't
    already have it.

The resulting features are simply `tf-idf counts <http://en.wikipedia.org/wiki/Tf%E2%80%93idf>`_ of the words found in the plot summaries for each movie. You can load them like this::

    >>> from mrec import load_sparse_matrix
    >>> features = load_sparse_matrix('npz','100k.features.npz')

and inspect the top few word counts for the first few items::

    >>> for i in xrange(3):
    ...     for tfidf,word in sorted(zip(features[i].data,features[i].indices),reverse=True)[:3]:
    ...         print '{0}\t{1}\t{2:.3f}'.format(i,word,tfidf)
    ...
    0   500 0.440
    0   549 0.340
    0   4   0.242
    1   311 0.412
    1   564 0.335
    1   549 0.243
    2   117 0.430
    2   286 0.427
    2   670 0.220

Now we can train a recommender in the usual way, specifying the features with the ``item_features``
and ``item_feature_format`` options::

    $ mrec_train -n4 --input_format tsv --train u.data.train.0 --outdir models --model warp --item_features 100k.features.npz --item_feature_format npz

Once this has finished (it will take a few minutes even on a single split of this small dataset)
you can use the recommender to make and evaluate predictions::

    $ mrec_predict --input_format tsv --test_input_format tsv --train u.data.train.0 --modeldir models --outdir recs --item_features 100k.features.npz --item_feature_format npz

After a few seconds you'll get the results as usual::

    WARP2MF(d=80,gamma=0.01,C=100.0)
    mrr            0.6008 +/- 0.0000
    prec@5         0.3650 +/- 0.0000
    prec@10        0.3221 +/- 0.0000
    prec@15        0.2915 +/- 0.0000
    prec@20        0.2699 +/- 0.0000

.. [1] Weston, J., Bengio, S., & Usunier, N. (2010). Large scale image annotation: learning to rank with joint word-image embeddings. Machine learning, 81(1), 21-35.
