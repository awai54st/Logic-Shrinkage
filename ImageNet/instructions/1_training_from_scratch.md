### Training from Scratch

`training-software` contains the source code for training.

We recommend using docker containers for reproducing our training results. 
For all our experiments with small-scale datasets, we used TF2 container (tag: 21.06-tf2-py3) provided by [Nvidia](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow).

The iterative training process is automated using a bash script `logic_shrinkage_experiment.sh`. To reproduce our key results with ImageNet using Bi-Real-18 network, please run

```
bash logic_shrinkage_experiment.sh
```

And you should expect a final accuracy of around 53.40%.
Output trained network is stored in `artefact/2_residuals.h5`, and training log in `artefact/training_log`.

Below is a list of parameters that users can use to configure training in `logic_shrinkage_experiment.sh`.

* __K__: number of inputs per LUT
* __mode__: 'baseline' (vanilla LUTNet) or 'ls' (logic shrinkage)
* __output_path__: directory containing the trained network
* __training_phase__: '1' means from scratch to pruned BNN, '2' means from pruned BNN to logic-shrunk LUTNet, and '0' means inference
* __activation_percentage__: LUT input sparsity i.e. &delta; in paper
* __pruning_percentage__: LUT node sparsity i.e. &theta; in paper

File `lutnet_logic_shrinkage.py` contains __batch sizes__ and __epoch numbers__ that users can finetune with.

### ImageNet Dataset Preparation

ImageNet 2012 dataset can be downloaded using the script `prepare_imagenet.py`.
