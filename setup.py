#!/usr/bin/python

from setuptools import setup, find_packages, Extension

from Cython.Distutils import build_ext
import numpy

import mrec

with open('README.rst') as f:
    long_description = f.read()

setup(packages=find_packages(),
      version=mrec.__version__,
      maintainer='Mark Levy',
      name='mrec',
      package_dir={'':'.'},
      maintainer_email='mark.levy@mendeley.com',
      description='mrec recommender systems library',
      long_description=long_description,
      url='https://github.com/mendeley/mrec',
      download_url='https://github.com/mendeley/mrec/tarball/master#egg=mrec-'+mrec.__version__,
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'License :: OSI Approved :: BSD License',
                   'Operating System :: Unix',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering',],
      install_requires=['numpy',
                        'scipy',
                        'scikit-learn',
                        'ipython <= 4.0.0',
                        'cython',
                        'psutil'],
      entry_points={
          'console_scripts':[
              'mrec_prepare = mrec.examples.prepare:main',
              'mrec_train = mrec.examples.train:main',
              'mrec_predict = mrec.examples.predict:main',
              'mrec_evaluate = mrec.examples.evaluate:main',
              'mrec_tune = mrec.examples.tune_slim:main',
              'mrec_convert = mrec.examples.convert:main',
              'mrec_factors = mrec.examples.factors:main',
          ]},
      cmdclass={'build_ext':build_ext},
      ext_modules=[Extension('warp_fast',
                             sources=['mrec/mf/model/warp_fast.pyx'],
                             include_dirs=[numpy.get_include()]),
                  ]
)
