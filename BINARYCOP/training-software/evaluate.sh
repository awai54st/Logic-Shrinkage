#!/bin/bash
K=4
mode='baseline' # baseline or ls
output_path='artefact_muBinaryCoP_baseline_rebnet_redo'
log_path='console_and_file.log'
custom_rand_seed=567
mkdir ${output_path}/training_log
training_phase='0'
activation_percentage='0'
pruning_percentage='0'
# TODO: CHANGE THIS LINE TO CHANGE DATASET
dataset='mu-BinaryCoP'
python top.py ${mode} ${K} ${pruning_percentage} ${activation_percentage} ${output_path} ${custom_rand_seed} ${training_phase} ${dataset}
#> ${output_path}/training_log/${mode}_${K}lutnet_evaluation_result_${pruning_percentage}_${activation_percentage}_${notes}_rep_${repeat}.txt