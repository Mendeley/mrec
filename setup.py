#!/usr/bin/python

from setuptools import setup, find_packages

setup(packages=find_packages(),
      version='0.1.2',
      maintainer='Mark Levy',
      name='mrec',
      package_dir={'':'.'},
      maintainer_email='mark.levy@mendeley.com',
      description='recommender systems library',
      install_requires=['numpy','scipy','scikit-learn','ipython'],
      entry_points={
          'console_scripts':[
              'mrec_prepare = mrec.examples.prepare:main',
              'mrec_train = mrec.examples.train:main',
              'mrec_predict = mrec.examples.predict:main',
              'mrec_evaluate = mrec.examples.evaluate:main',
              'mrec_tune = mrec.examples.tune_slim:main',
          ]})
