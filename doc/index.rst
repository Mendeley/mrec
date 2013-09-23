================================
Welcome to mrec's documentation!
================================

Introduction
------------
`mrec` is a Python package developed at `Mendeley <http://www.mendeley.com>`_ to support recommender systems development and evaluation.  The package currently focuses on item similarity methods and experimental evaluation.

Why another package when there are already some really good software projects implementing recommender systems?

`mrec` tries to fill two small gaps in the current landscape, firstly by supplying
simple tools for consistent and reproducible evaluation, and secondly by offering examples
of how to use IPython.parallel to run the same code either on the cores of a single machine
or on a cluster.  The combination of IPython and scientific Python libraries is very powerful,
but there are still rather few examples around that show how to get it to work in practice.

Highlights:

- a (relatively) efficient implementation of the SLIM item similarity method.
- utilities to train models and make recommendations in parallel using IPython.
- utilities to prepare datasets and compute quality metrics.
 
Contents
--------

.. toctree::
    :maxdepth: 1

    quickstart
    preparation
    training
    evaluation
    aws
    api


