#!/bin/bash

K=4
mode='ls' # baseline or ls
output_path='artefact'
custom_rand_seed=567

mkdir ${output_path}
mkdir ${output_path}/training_log
#rm ${output_path}/dummy.h5

training_phase='2'
activation_percentage='0.75'
notes='3iter_retrain'

pruning_percentage='0.30'

#for repeat in '0' '1' '2' '3' '4'
for repeat in '0'
do
for training_phase in '1' '2'
do
#cp ${output_path}/training_log/ls_$((K-num_act_pruned))0lutnet_${pruning_percentage}_phase_1.h5 ${output_path}/2_residuals.h5
python top.py ${mode} ${K} ${pruning_percentage} ${activation_percentage} ${output_path} ${custom_rand_seed} ${training_phase} > ${output_path}/training_log/${mode}_${K}lutnet_${pruning_percentage}_${activation_percentage}_${notes}_rep_${repeat}_phase_${training_phase}.txt
done
done

