#!/usr/bin/env sh

FOLDS=4

for fold in $(seq 0 $FOLDS)
do
    ./ir_metrics.py -s ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/results/similarity_reranking.json \
        -t ~/Datasets/DHC/variations/new/var_1/targets/case_advs.json \
        -l ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/logs/ \
        -o ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/metrics \
        -n hard
done

for fold in $(seq 0 $FOLDS)
do
    ./ir_metrics.py -s ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/results/similarity_reranking.json \
        -t ~/Datasets/DHC/variations/new/var_1/targets/case_winners.json \
        -l ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/logs/ \
        -o ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/metrics \
        -n harder
done
