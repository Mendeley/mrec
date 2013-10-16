"""
Convert sparse matrix from one file format to another.
"""

import os
import subprocess

def tsv2mtx(infile,outfile):
    num_users,num_items,nnz = 0,0,0
    for line in open(infile):
        u,i,v = line.strip().split()
        u = int(u)
        i = int(i)
        if u > num_users:
            num_users = u
        if i > num_items:
            num_items = i
        nnz += 1
    headerfile = outfile+'.header'
    with open(headerfile,'w') as header:
        print >>header,'%%MatrixMarket matrix coordinate real general'
        print >>header,'{0} {1} {2}'.format(num_users,num_items,nnz)
    subprocess.check_call(['cat',headerfile,infile],stdout=open(outfile,'w'))
    subprocess.check_call(['rm',headerfile])

def main():
    from optparse import OptionParser

    from mrec import load_sparse_matrix, save_sparse_matrix

    parser = OptionParser()
    parser.add_option('--input_format',dest='input_format',help='format of input dataset tsv | csv | mm (matrixmarket) | csr (scipy.sparse.csr_matrix) | fsm (mrec.sparse.fast_sparse_matrix)')
    parser.add_option('--input',dest='input',help='filepath to input')
    parser.add_option('--output_format',dest='output_format',help='format of output dataset(s) tsv | csv | mm (matrixmarket) | csr (scipy.sparse.csr_matrix) | fsm (mrec.sparse.fast_sparse_matrix)')
    parser.add_option('--output',dest='output',help='filepath for output')

    (opts,args) = parser.parse_args()
    if not opts.input or not opts.output or not opts.input_format or not opts.output_format:
        parser.print_help()
        raise SystemExit

    if opts.output_format == opts.input_format:
        raise SystemExit('input and output format are the same, not doing anything')

    if opts.input_format == 'tsv' and opts.output_format == 'mm':
        # we can do this without loading the data
        tsv2mtx(opts.input,opts.output)
    else:
        data = load_sparse_matrix(opts.input_format,opts.input)
        save_sparse_matrix(data,opts.output_format,opts.output)

if __name__ == '__main__':
    main()

