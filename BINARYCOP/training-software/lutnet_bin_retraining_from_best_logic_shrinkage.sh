#!/bin/bash

# TODO: CHANGE THIS LINE TO CHANGE DATASET
dataset='mu-BinaryCoP'

K=4
mode='lutnet_bin_retraining_from_best_logic_shrinkage'
pruning_percentage="0.60" # node sparsity θ
activation_percentage="0.875" # LUT input sparsity i.e. δ in paper, 𝛿 = 0 is the LUTNet design and 𝛿 > 0 is the Logic-Shrinkage design
best_lutnet_output_path="artefact_mu-BinaryCoP_bin_retraining_from_best_logic_shrunk_0.60_0.875"
custom_rand_seed=567
notes="3iter_retrain"

# for repeat in '1' '2' '3' '4' #'5'
for repeat in '1' '2' '3' '4' '5' '6' '7' '8'
do
    output_path="artefact_${dataset}_${mode}_${pruning_percentage}_${activation_percentage}_rep_${repeat}"
    if [[ -d ${output_path} ]]
    then
        echo "${output_path} exists on your filesystem."
    else
        mkdir ${output_path}
        mkdir ${output_path}/training_log
        cp ${best_lutnet_output_path}_2_residuals.h5 ${output_path}/2_residuals.h5
        for training_phase in '2'
            do
            python top.py ${mode} ${K} ${pruning_percentage} ${activation_percentage} ${output_path} ${custom_rand_seed} ${training_phase} ${dataset} > ${output_path}/training_log/${mode}_${K}lutnet_${pruning_percentage}_${activation_percentage}_${notes}_phase_${training_phase}_rep_${repeat}.txt
            cp console_and_file.log ${output_path}/training_log/console_and_file_${dataset}_${mode}_${pruning_percentage}_${activation_percentage}_phase_${training_phase}_rep_${repeat}.log
            rm -r console_and_file.log
        done
    fi
done