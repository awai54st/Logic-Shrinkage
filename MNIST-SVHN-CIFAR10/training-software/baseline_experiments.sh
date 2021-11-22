#!/bin/bash

K=4
mode='baseline' # baseline or ls
output_path='artefact_baseline'
custom_rand_seed=567

mkdir ${output_path}
mkdir ${output_path}/training_log
#rm ${output_path}/dummy.h5

training_phase='1'
activation_percentage='0.75'
#repeat='0'
notes='3iter_retrain'

##for pruning_percentage in '0.80' '0.85' '0.90' '0.95' '0.98' '0.985' '0.99' '0.995' '0.9975'
#for pruning_percentage in '0.975'
#do
##for training_phase in '1' '2'
##do
#	python top.py ${mode} ${K} ${num_act_pruned} ${pruning_percentage} ${output_path} ${custom_rand_seed} ${training_phase} > ${output_path}/training_log/${mode}_$((K-num_act_pruned))0lutnet_${pruning_percentage}_${activation_percentage}_phase${training_phase}.txt
##done
#done

pruning_percentage='0.94'

#for repeat in '0' '1' '2' '3' '4'
for repeat in '0'
do
for training_phase in '1' '2'
do
#cp debug_cifar10/training_log/ls_${K}lutnet_${pruning_percentage}_phase_1.h5 ${output_path}/2_residuals.h5
python top.py ${mode} ${K} ${num_act_pruned} ${pruning_percentage} ${activation_percentage} ${output_path} ${custom_rand_seed} ${training_phase} > ${output_path}/training_log/${mode}_${K}lutnet_${pruning_percentage}_${activation_percentage}_${notes}_rep_${repeat}.txt
done
done

