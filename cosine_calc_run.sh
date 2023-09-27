#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./cosine_calc.py -d /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/embeddings/train_rep \
        -q /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/embeddings/test_rep/ \
        -o /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/results/ \
        -l /home/workboots/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/logs/
done

