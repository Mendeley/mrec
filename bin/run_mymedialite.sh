#!/bin/bash

MYMEDIALITE_HOME=$1
TRAIN_GLOB=$2

mkdir -p wrmf_recs

for TRAIN in $TRAIN_GLOB
do
    recspath=wrmf_recs/`basename $TRAIN`.recs

    $MYMEDIALITE_HOME/bin/item_recommendation --training-file $TRAIN \
       --recommender WRMF \
       --predict-items-number 20 \
       --prediction-file $recspath.tmp

    python -c \
"import sys
for line in sys.stdin:
    u,z = line.strip().split()
    recs = eval(z.replace(',','),(').replace(':',',').replace('[','[(').replace(']',')]'))
    for i,v in recs:
        print '{0}\t{1}\t{2}'.format(u,i,v)" \
        < $recspath.tmp \
        > $recspath.tsv

    rm $recspath.tmp
done

mrec_evaluate --input_format tsv --test_input_format tsv \
    --train "$TRAIN_GLOB" \
    --recsdir wrmf_recs \
    --description "wrmf"
