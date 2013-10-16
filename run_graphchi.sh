#!/bin/bash

GRAPHCHI_HOME=/home/mark/dev/hg/graphchi
HERE=`pwd`
TRAIN_GLOB=$1

mkdir -p mm_factors

for TRAIN in $TRAIN_GLOB
do
    cd $HERE
    mrec_convert --input_format tsv \
    --input $TRAIN \
    --output_format mm \
    --output $TRAIN.mm

    cd $GRAPHCHI_HOME
    ./toolkits/collaborative_filtering/climf \
        --training=$HERE/$TRAIN.mm  \
        --binary_relevance_thresh=1 --sgd_lambda=0.1 --sgd_gamma=0.0001 --max_iter=10 --quiet=1
    rm -r $HERE/$TRAIN.mm.* $HERE/$TRAIN.mm_degs.bin

    cd $HERE
    mrec_factors --factor_format mm \
        --user_factors $TRAIN.mm_U.mm \
        --item_factors $TRAIN.mm_V.mm \
        --train $TRAIN \
        --outdir als_models \
        --description als
    mv $HERE/$TRAIN.mm* mm_factors/
done

mrec_predict -n4 --input_format tsv --test_input_format tsv \
    --train "$TRAIN_GLOB" \
    --modeldir als_models \
    --outdir als_recs
