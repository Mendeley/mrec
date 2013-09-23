"""
File naming conventions for test and similarity matrix files:

* training files must contain 'train' in their filename.
* the corresponding test files must have the same filepaths,
  but with 'test' in place of 'train' in their filenames.
* the similarity matrix generated for a training file
  will be called 'sims.tsv' and will be output into a directory
  based on the training filename.

"""

import os

def get_testfile(trainfile):
    filename = os.path.basename(trainfile)
    return os.path.join(os.path.dirname(trainfile),filename.replace('train','test'))

def get_simsdir(trainfile,outdir):
    filename = os.path.basename(trainfile)
    return os.path.join(outdir,'{0}-sims'.format(filename))

def get_recsdir(trainfile,outdir):
    filename = os.path.basename(trainfile)
    return os.path.join(outdir,'{0}-recs'.format(filename))

def get_simsfile(trainfile,outdir):
    filename = os.path.basename(trainfile)
    return os.path.join(outdir,'{0}.sims.tsv'.format(filename))

def get_recsfile(trainfile,outdir):
    filename = os.path.basename(trainfile)
    return os.path.join(outdir,'{0}.recs.tsv'.format(filename))

def get_modelfile(trainfile,outdir):
    filename = os.path.basename(trainfile)
    return os.path.join(outdir,'{0}.model.npz'.format(filename))

def get_sortedfile(infile,outdir):
    filename = os.path.basename(infile)
    return os.path.join(outdir,'{0}.sorted'.format(filename))

def get_splitfile(infile,outdir,split_type,i):
    filename = os.path.basename(infile)
    return os.path.join(outdir,'{0}.{1}.{2}'.format(filename,split_type,i))
