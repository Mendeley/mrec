================================
mrec recommender systems library
================================

Introduction
------------
`mrec` is a Python package developed at `Mendeley <http://www.mendeley.com>`_ to support recommender systems development and evaluation.  The package currently focuses on item similarity and other methods that work well on implicit feedback, and on experimental evaluation.

Why another package when there are already some really good software projects implementing recommender systems?

`mrec` tries to fill two small gaps in the current landscape, firstly by supplying
simple tools for consistent and reproducible evaluation, and secondly by offering examples
of how to use IPython.parallel to run the same code either on the cores of a single machine
or on a cluster.  The combination of IPython and scientific Python libraries is very powerful,
but there are still rather few examples around that show how to get it to work in practice.

Highlights:

- a (relatively) efficient implementation of the SLIM item similarity method [1]_.
- an implementation of Hu, Koren & Volinsky's WRMF weighted matrix factorization for implicit feedback [2]_.
- a matrix factorization model that optimizes the Weighted Approximately Ranked Pairwise (WARP) ranking loss [3]_.
- a hybrid model optimizing the WARP loss for a ranking based jointly on a user-item matrix and on content features for each item.
- utilities to train models and make recommendations in parallel using IPython.
- utilities to prepare datasets and compute quality metrics.

Documentation for mrec can be found at http://mendeley.github.io/mrec.

The source code is available at https://github.com/mendeley/mrec.

`mrec` implements the SLIM recommender described in [1]_.  Please cite this paper if you 
use `mrec` in your research.

References
----------
.. [1] Mark Levy, Kris Jack (2013). Efficient Top-N Recommendation by Linear Regression. In Large Scale Recommender Systems Workshop in RecSys'13.
.. [2] Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative filtering for implicit feedback datasets. In IEEE ICDM'08.
.. [3] Weston, J., Bengio, S., & Usunier, N. (2010). Large scale image annotation: learning to rank with joint word-image embeddings. Machine learning, 81(1), 21-35.
