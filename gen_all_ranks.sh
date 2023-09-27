#!/usr/bin/env sh

FOLDS=4

# Generate similarity ranking and re-ranking for all provided FOLDS

for fold in $(seq 0 $FOLDS)
do
    python calculate_ranks.py -d ~/Datasets/DHC/variations/new/var_1/cross_val/5_fold/fold_$fold/adv_split_info.json \
        -ct ~/Datasets/DHC/variations/new/var_1/targets/case_advs.json \
        -s ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/results/scores.json \
        -o ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/results/ \
        -l ~/Results/advocate_recommendation/new/exp_3.1/cross_val/5_fold/fold_$fold/logs/
done
