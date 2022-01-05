# Logic Shrinkage: Learned FPGA Netlist Sparsity for Efficient Neural Network Inference

In this repository we present Logic Shrinkage, a fine-grained netlist pruning methodology enabling K to be automatically learned for every K-input LUT in a neural network targeted for FPGA inference.

Logic Shrinkage is developed from LUTNet, an end-to-end hardware-software framework for the construction of area-efficient FPGA-based neural network accelerators. Please checkout our previous [repository](https://github.com/awai54st/LUTNet) and publications ([conference paper](https://arxiv.org/abs/2112.02346) and/or [journal article](https://arxiv.org/abs/1910.12625)) on LUTNet.

## Repo organisation

We separated ImageNet from small-scale networks due to slight differences in training environments.
Below is an overview of sub-directories.

* __instructions__: detailed instructions on reproducing our results from scratch
* __lutnet/h5py-2-hls__: script which converts pretrained network (.h5) into HLS header files (.h) and LUT array RTLs (.v)
* __lutnet/src/library__: HLS library
* __lutnet/src/network__: Vivado synthesis code
* __lutnet/src/network/LUTNET_c6__ (or __LUTNET_MNIST__ if dataset is MNIST, __LUTNET_IMAGENET__ if ImageNet): HLS project
* __lutnet/src/network/vivado_out__: Vivado synthesis output project
* __training-software__: Tensorflow-based training project
* __training-software/artefact/2_residuals.h5__: Training output

## Prerequisites

For training Logic Shrinkage, you should have the following packages installed:
* Keras (v2)
* TensorFlow

We recommend using [Nvidia's docker containers for TensorFlow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow) to setup training environments.
We developed the project using __21.06-tf1-py3__ for __MNIST-CIFAR-SVHN__, and __21.06-tf2-py3__ for __ImageNet__, respectively.

For hardware synthesis, we developed and tested the project with Vivado (+ HLS) 2016.3. 
Newer versions of Vivado HLS do not work with our project. 
In newer versions of Vivado HLS, loop unrolling factors are limited, reducing the area-efficiency advantage of LUTNet.

## Results

<table>
  <tr>
    <td>Dataset</td>
    <td>Model</td>
    <td>Target layer</td>
    <td>Top-1 Accuracy (%)</td>
    <td>LUT</td>
    <td>FPS (Target layer only)</td>
  </tr>
  <tr>
    <td>MNIST</td>
    <td>Multilayer Perceptron</td>
    <td>All dense layers</td>
    <td>97.47</td>
    <td>63928</td>
    <td>200M</td>
  </tr>
  <tr>
    <td>SVHN</td>
    <td>CNV</td>
    <td>Largest conv layer</td>
    <td>96.25</td>
    <td>179236</td>
    <td>200M</td>
  </tr>
  <tr>
    <td>CIFAR-10</td>
    <td>CNV</td>
    <td>Largest conv layer</td>
    <td>84.74</td>
    <td>220060</td>
    <td>200M</td>
  </tr>
  <tr>
    <td>ImageNet</td>
    <td>Bi-Real-18</td>
    <td>Largest conv layer</td>
    <td>53.40</td>
    <td>690357</td>
    <td>5.56M</td>
  </tr>
</table>

## Citation

If you make use of this code, please acknowledge us by citing our conference papers ([FCCM'19](https://arxiv.org/abs/1904.00938), [FPGA'22](https://arxiv.org/abs/2112.02346)) and/or [journal article](https://arxiv.org/abs/1910.12625):

    @inproceedings{lutnet_fccm,
		author={Wang, Erwei and Davis, James J. and Cheung, Peter Y. K. and Constantinides, George A.},
		title={{LUTNet}: Rethinking Inference in {FPGA} Soft Logic},
		booktitle={IEEE International Symposium on Field-Programmable Custom Computing Machines},
		year={2019}
    }

	@article{lutnet_tc,
		author={Wang, Erwei and Davis, James J. and Cheung, Peter Y. K. and Constantinides, George A.},
		title={{LUTNet}: Learning {FPGA} Configurations for Highly Efficient Neural Network Inference},
		journal={IEEE Transactions on Computers},
		year={2020},
		volume = {69},
		number = {12},
		issn = {1557-9956},
		pages = {1795-1808},
		doi = {10.1109/TC.2020.2978817}
	}
	
	@inproceedings{lutnet_ls_fpga,
		author={Wang, Erwei and Davis, James J. and Stavrou, Georgios-Ilias and Cheung, Peter Y. K. and Constantinides, George A. and Abdelfattah, Mohamed},
		title={Logic Shrinkage: Learned {FPGA} Netlist Sparsity for Efficient Neural Network Inference},
		booktitle={ACM/SIGDA International Symposium on Field-Programmable Gate Arrays},
		year={2022},
		note={to appear}
    }
