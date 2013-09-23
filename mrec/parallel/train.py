"""
Training task to run on an ipython engine.
"""

def run(task):

    # import modules required by engine
    import os
    import subprocess
    from mrec import load_fast_sparse_matrix

    model,input_format,trainfile,outdir,start,end,max_similar_items = task

    # initialise the model
    dataset = load_fast_sparse_matrix(input_format,trainfile)
    model.init(dataset)
    if hasattr(model,'similarity_matrix'):
        # clear out any existing similarity matrix
        model.similarity_matrix = None

    # write sims directly to file as we compute them
    outfile = os.path.join(outdir,'sims.{0}-{1}.tsv'.format(start,end))
    out = open(outfile,'w')
    for j in xrange(start,end):
        w = model.get_similar_items(j,max_similar_items=max_similar_items)
        for k,v in w:
            print >>out,'{0}\t{1}\t{2}'.format(j+1,k+1,v)  # write as 1-indexed
    out.close()

    # record success
    cmd = ['touch',os.path.join(outdir,'{0}-{1}.SUCCESS'.format(start,end))]
    subprocess.check_call(cmd)

    # return the range that we've processed
    return (start,end)
