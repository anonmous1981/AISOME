#!/usr/bin/env zsh

FOLDS=4
THRESHOLDS=(1.0 0.80)

for i in $(seq 0 $FOLDS)
do
    ./map.py \
    -d ~/Results/advocate_recommendation/exp_4/metrics/bert_base_uncased_pred_DHC_SC_new_facts_fold_$i/similarity_reranking.json \
    -i ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$i/adv_case_splits.json \
    -t ~/Datasets/DHC/common/case_winners.json \
    -n map_harder

    ./map.py \
    -d ~/Results/advocate_recommendation/exp_4/metrics/bert_base_uncased_pred_DHC_SC_new_facts_end_tokens_fold_$i/similarity_reranking.json \
    -i ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$i/adv_case_splits.json \
    -t ~/Datasets/DHC/common/case_winners.json \
    -n map_harder
done


# for score in ${THRESHOLDS[@]}
# do
    # for i in $(seq 0 $FOLDS)
    # do
        # ./map.py \
        # -d ~/Results/advocate_recommendation/exp_4/metrics/bert_base_uncased_pred_DHC_SC_new_facts_fold_$i/similarity_reranking.json \
        # -i ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$i/adv_case_splits.json \
        # -t ~/Datasets/DHC/common/case_advs.json \
        # -n map_$score \
        # -c ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
        # -ac ~/Datasets/DHC/variations/var_3/targets/adv_ipc_charges.json \
        # -th $score

        # ./map.py \
        # -d ~/Results/advocate_recommendation/exp_4/metrics/bert_base_uncased_pred_DHC_SC_new_facts_end_tokens_fold_$i/similarity_reranking.json \
        # -i ~/Datasets/DHC/variations/var_2/data/ipc_data/cross_val/20_fold/fold_$i/adv_case_splits.json \
        # -t ~/Datasets/DHC/common/case_advs.json \
        # -n map_$score \
        # -c ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json \
        # -ac ~/Datasets/DHC/variations/var_3/targets/adv_ipc_charges.json \
        # -th $score
    # done
# done
