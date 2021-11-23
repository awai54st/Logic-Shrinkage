### Vivado Synthesis

`lutnet/src` contains our source code for implementing the network using Vivado 16.3. (Note: version 17.1+ will fail due to it limiting the maximum unrolling factors in HLS)

Below lists the procedure for HLS, synthesis and implementation.

* Copy the `.h` files from `h5py-2-hls/codegen_output` to `src/network/IMAGENET/hw/`
* Run `bash lutnet_synthesis_script_part1.sh` to perform HLS
* Copy the `.v` files from `h5py-2-hls/codegen_output` to `src/network/LUTNet_IMAGENET/sol1/syn/verilog/` (`cp lutnet/h5py-2-hls/codegen_output/LUTARRAY_bX_0.v lutnet/src/network/LUTNet_IMAGENET/sol1/syn/verilog/LUTARRAY_bX.v`, where `X` &isin; {0, 1})  Overwrite files if prompted
* Run `bash lutnet_synthesis_script_part2.sh` to perform synthesis and implementation
* Output Vivado project is stored in `src/network/vivado_out`
