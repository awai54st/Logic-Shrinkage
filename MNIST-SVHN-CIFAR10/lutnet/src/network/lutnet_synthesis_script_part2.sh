#!/bin/bash
task_name="CNV"

if [[ ${task_name} == "MNIST" ]]
then
  vivado_proj_dir="LUTNET_MNIST"
elif [[ ${task_name} == "CNV" ]]
then
  vivado_proj_dir="LUTNET_c6"
fi
vivado_hls -f hls-export.tcl
rm -r ip_catalog/ip/
cp -r ${vivado_proj_dir}/sol1/impl/ip ip_catalog/
bash vivado_syn.sh


