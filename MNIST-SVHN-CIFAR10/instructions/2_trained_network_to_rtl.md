### From Trained TF Network (.h5) to HLS header (.h) and RTL (.v)

`lutnet/h5py-2-hls` contains the source code for converting trained LUTNets to RTLs.

Copy trained network (.h5 format) into lutnet/h5py-2-hls/[network] directory, rename it to `pretrained_network_4lut.h5`, then run

```
python h52header_4lut_sparse.py
```

This script will convert the network into HLS headers (.h) and LUT array rtl files (.v), which are stored in `lutnet/h5py-2-hls/codegen_output`.
