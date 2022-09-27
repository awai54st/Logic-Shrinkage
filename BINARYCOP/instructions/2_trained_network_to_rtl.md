## From Trained TF Network (.h5) to HLS header (.h) and RTL (.v)

`lutnet/h5py-2-hls` contains the source code for converting trained LUTNets to RTLs.

Copy trained network (.h5 format) into lutnet/h5py-2-hls/ directory, rename it to `pretrained_network.h5`.

### Pretrained network is a tiled unpruned rebnet architecture

If the pretrained network is a tiled unpruned rebnet architecture for the mu-CNV network model classifying BinaryCoP, run

```
python h52header_tiled_unpruned_rebnet
```

This script will convert the network into HLS headers (.h), which are stored in `lutnet/h5py-2-hls/pretrained_network`.

### Pretrained network is an unrolled unpruned rebnet architecture

If the pretrained network is an unrolled unpruned rebnet architecture for the mu-CNV network model classifying BinaryCoP, 

* Edit `lutnet/h5py-2-hls/h52header_tiled_unpruned_rebnet.py` line 105 to set ls_layer_id = 5
* Run

```
python h52header_tiled_unpruned_rebnet
```

* Run

```
python h52header_unrolled_unpruned_rebnet_conv3_1
```

This script will convert the network into HLS headers (.h) and XNOR array rtl files (.v), which are stored in `lutnet/h5py-2-hls/pretrained_network`.

### Pretrained network is an unrolled pruned rebnet architecture

If the pretrained network is an unrolled pruned rebnet architecture for the mu-CNV network model classifying BinaryCoP, 

* Edit `lutnet/h5py-2-hls/h52header_tiled_unpruned_rebnet.py` line 105 to set ls_layer_id = 5
* Run

```
python h52header_tiled_unpruned_rebnet
```

* Run

```
python h52header_unrolled_pruned_rebnet_conv3_1
```

This script will convert the network into HLS headers (.h) and XNOR array rtl files (.v), which are stored in `lutnet/h5py-2-hls/pretrained_network`.

### Pretrained network is an unrolled pruned LUTNet/Logic-shrunk architecture

If the pretrained network is an unrolled pruned LUTNet/Logic-shrunk for the mu-CNV network model classifying BinaryCoP, 

* Edit `lutnet/h5py-2-hls/h52header_tiled_unpruned_rebnet.py` line 105 to set ls_layer_id = 5
* Run

```
python h52header_tiled_unpruned_rebnet
```

* Run

```
python h52header_4lut_lutnet_conv3_1
```

This script will convert the network into HLS headers (.h) and LUT array rtl files (.v), which are stored in `lutnet/h5py-2-hls/pretrained_network`.