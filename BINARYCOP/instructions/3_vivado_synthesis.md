## Vivado Synthesis

`lutnet/src` contains our source code for implementing the network using Vivado 16.3. (Note: version 17.1+ will fail due to it limiting the maximum unrolling factors in HLS)

Below lists the procedure for HLS, synthesis and implementation.

### HLS synthesis

* Copy all the `.h` files from `h5py-2-hls/pretrained_network/hw/` to `src/network/{network_name}/hw/`
* Run `./make-hw.sh {network_name} {platform} s` to launch Vivado HLS synthesis where:
	- network_name can be `mubincop-logic_shrunk`, `mubincop-pruned_unrolled_rebnet`, `mubincop-tiled_unpruned_rebnet` or `mubincop-unpruned_unrolled_rebnet`;
	- platform is `pynq` or `kintex` (simulation only);

* Output Vivado HLS project is stored in `src/network/output/hls/{network_name}/`

Additional step for unrolled network architectures (unrolled unpruned ReBNet, unrolled pruned ReBNet, unrolled pruned LUTNet/Logic-shrunk):
* Copy the `.v` files from `h5py-2-hls/pretrained_network` to `src/network/output/hls/{network_name}/sol1/syn/verilog/` (overwrite files if prompted)

### HLS IP export

* Run `./make-hw.sh {network_name} {platform} e` to launch Vivado HLS IP export where:
	- network_name can be `mubincop-logic_shrunk`, `mubincop-pruned_unrolled_rebnet`, `mubincop-tiled_unpruned_rebnet` or `mubincop-unpruned_unrolled_rebnet`;
	- platform is `pynq` or `kintex` (simulation only);

### Synthesis and implementation

* Run `./make-hw.sh {network_name} {platform} b` to launch the Vivado project where:
	- network_name can be `mubincop-logic_shrunk`, `mubincop-pruned_unrolled_rebnet`, `mubincop-tiled_unpruned_rebnet` or `mubincop-unpruned_unrolled_rebnet`;
	- platform is `pynq` or `kintex` (simulation only);

* Output Vivado project is stored in `src/network/output/vivado/{network_name}/`