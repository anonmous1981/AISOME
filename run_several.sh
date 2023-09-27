#!/usr/bin/sh

# Script to run several models one after another. All results will be placed in 
# experiments under model_states and metrics in directories according to the 
# name given to each model

# New Facts, DHC and SC
python train.py -d ~/Datasets/DHC/variations/var_3/data/ipc_data/cross_val/5_fold/fold_0/ ~/Datasets/SC/variations/var_1.1/data/ipc_data/fact_sentences \
    -t ~/Datasets/DHC/variations/var_3/targets/ipc_case_offences.json ~/Datasets/SC/variations/var_1.1/targets/ipc_case_offences.json \
    -n han_pred_DHC_SC_new_facts -p params_word2vec_200.json \
    -ul ~/Datasets/DHC/variations/var_3/targets/unique_labels.txt \
    -bm bert-base-uncased


