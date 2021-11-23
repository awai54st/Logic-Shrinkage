### Training from Scratch

`training-software` contains the source code for training.

We recommend using docker containers for reproducing our training results. 
For all our experiments with small-scale datasets, we used TF1 container (tag: 21.06-tf1-py3) provided by [Nvidia](https://ngc.nvidia.com/catalog/containers/nvidia:tensorflow).

The iterative training process is automated using a bash script `logic_shrinkage_experiment.sh`. To reproduce our key results with CIFAR-10 using CNV network, please run

```
bash logic_shrinkage_experiment.sh
```

And you should expect a final accuracy of around 84.74%.
Output trained network is stored in `artefact/2_residuals.h5`, and training log in `artefact/training_log`.

Below is a list of parameters that users can use to configure training in logic_shrinkage_experiment.sh.

* __K__: number of inputs per LUT
* __mode__: 'baseline' (vanilla LUTNet) or 'ls' (logic shrinkage)
* __output_path__: directory containing the trained network
* __training_phase__: '1' means from scratch to pruned BNN, '2' means from pruned BNN to logic-shrunk LUTNet, and '0' means inference
* __activation_percentage__: LUT input sparsity i.e. &delta; in paper
* __pruning_percentage__: LUT node sparsity i.e. &theta; in paper

File `lutnet_logic_shrinkage.py` contains __batch sizes__ and __epoch numbers__ that users can finetune with.

---

### Advanced: Changing Network

To switch from CIFAR-10 to MNIST or SVHN, please do the following:

* Edit `top.py` line 15
* Edit `lutnet_init.py` line 9-12
* Edit `logic_shrinkage.py` line 30-36
* Edit `Binary.py` line 48


### Special instructions on MNIST

For each LUTNet layer, there should be a pair of LUT array rtl files corresponding to activation bit 0 and 1.
Our MLP experiment classifying MNIST contains multiple LUTNet layers.
Vivado HLS generates random rtl source file names for LUT arrays.
Thus, the user has to make sure that the file names match for each HLS run.
The HLS project given in this package has the following file name mapping.

| `h5py-2-hls/codegen_output`	|	`src/network/LUTNet_MNIST/sol1/syn/verilog` |
| --------------------------- | ------------------------------------------- |
| LUTARRAY_b0_MNIST_2.v       | LUTARRAY_b0_MNIST.v                         |
| LUTARRAY_b0_MNIST_1.v	      | LUTARRAY_b0_MNIST_1.v                       |
| LUTARRAY_b0_MNIST_3.v       | LUTARRAY_b0_MNIST_2.v                       |
| LUTARRAY_b0_MNIST_4.v	      | LUTARRAY_b0_MNIST_3.v                       |
| LUTARRAY_b1_MNIST_4.v	      | LUTARRAY_b1_MNIST.v                         |
| LUTARRAY_b1_MNIST_3.v	      | LUTARRAY_b1_MNIST_1.v                       |
| LUTARRAY_b1_MNIST_1.v	      | LUTARRAY_b1_MNIST_2.v                       |
| LUTARRAY_b1_MNIST_2.v	      | LUTARRAY_b1_MNIST_3.v                       |

If a new HLS run is executed and the rtl file names are randomized again, go to `src/network/LUTNet_c6/sol1/syn/verilog/LUTARRAY_bX_MNIST_X.v` and check for the layer id in binary format.
These ids indicate which layers they correspond to.
