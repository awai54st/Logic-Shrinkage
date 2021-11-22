#!/bin/bash

K=4
mode='ls' # baseline or ls
output_path='artefact'

training_phase='0'
activation_percentage='0'

pruning_percentage='0'

mkdir ${output_path}/training_log

python top.py ${mode} ${K} ${pruning_percentage} ${activation_percentage} ${output_path} ${custom_rand_seed} ${training_phase} > ${output_path}/training_log/${mode}_${K}lutnet_evaluation_result.txt

