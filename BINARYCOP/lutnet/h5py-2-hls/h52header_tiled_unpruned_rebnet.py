################################################################################
# Imports
################################################################################

import h5py
import numpy as np
import os
from finnthesizer import *

################################################################################
# Pack model parameters in lists from H5 file
################################################################################

# load the parameters from h5 file into numpy arrays to feed the finnsynthesizer
def modelParametersList(h5_file_name, last_conv_layer_id, last_layer_id, ls_layer_id):
	print("\nLoading the pretrained parameters...")

	bl = h5py.File(h5_file_name, 'r')
	
	# init model parameter lists
	batch_norm_eps=1e-4
	weights = []
	gammas = []
	means = []
	pruning_masks = []
	rand_maps = []
	bn_betas = []
	bn_gammas = []
	bn_means = []
	bn_inv_stds = []

	for layer_id in range(1,last_layer_id+1):
		if layer_id != ls_layer_id:
			# Extract parameters for each layer

			if layer_id in range(1,last_conv_layer_id+1):
				# conv layer
				layer_name = "binary_conv_"+str(layer_id)
				bl_w1 = np.transpose(np.array(bl["model_weights"][layer_name][layer_name]["Variable_1:0"]), (3,2,0,1))

			else:
				# FC layer
				layer_name = "binary_dense_"+str(layer_id-last_conv_layer_id)
				bl_w1 = np.array(bl["model_weights"][layer_name][layer_name]["Variable_1:0"])

			bl_pruning_mask = np.array(bl["model_weights"][layer_name][layer_name]["pruning_mask:0"])
			#bl_w1 = np.transpose(np.array(bl["model_weights"][layer_name][layer_name]["Variable_1:0"]).reshape(bl_pruning_mask.shape))
			print("bl_w1", bl_w1.shape)
			bl_rand_map = np.array(bl["model_weights"][layer_name][layer_name]["rand_map_0:0"])
			bl_gamma = np.array(bl["model_weights"][layer_name][layer_name]["Variable:0"])
			print("bl_gamma", bl_gamma)
			
			bn_layer_name = "batch_normalization_"+str(layer_id)
			bl_bn_beta = np.array(bl["model_weights"][bn_layer_name][bn_layer_name]["beta:0"])
			bl_bn_gamma = np.array(bl["model_weights"][bn_layer_name][bn_layer_name]["gamma:0"])
			bl_bn_mean = np.array(bl["model_weights"][bn_layer_name][bn_layer_name]["moving_mean:0"])
			bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"][bn_layer_name][bn_layer_name]["moving_variance:0"])+batch_norm_eps)
			
			if layer_id != last_layer_id:
				rs_layer_name = "residual_sign_"+str(layer_id)
				bl_means = bl["model_weights"][rs_layer_name][rs_layer_name]["means:0"]
		
		else:
			if layer_id in range(1,last_conv_layer_id+1):
				# conv layer
				layer_name = "binary_conv_"+str(layer_id)
				print(layer_name)
			else:
				# FC layer
				layer_name = "binary_dense_"+str(layer_id-last_conv_layer_id)
				print(layer_name)	
		
			bl_pruning_mask = []
			bl_w1 = []
			bl_rand_map = []
			bl_gamma = []
			bn_layer_name = []
			bl_bn_beta = []
			bl_pruning_mask = []
			bl_bn_gamma = []
			bl_bn_inv_std = []
			if layer_id != last_layer_id:
				rs_layer_name = "residual_sign_"+str(layer_id)
				bl_means = bl["model_weights"][rs_layer_name][rs_layer_name]["means:0"]
				
		# Add the layer parameters to returned lists
		w_lut = [bl_w1]
		weights.extend([w_lut])
		gammas.extend([bl_gamma])
		pruning_masks.extend([bl_pruning_mask])
		rand_maps.extend([bl_rand_map])
		if layer_id != last_layer_id:
			means.extend([bl_means])
		bn_betas.extend([bl_bn_beta])
		bn_gammas.extend([bl_bn_gamma])
		bn_means.extend([bl_bn_mean])
		bn_inv_stds.extend([bl_bn_inv_std])

	return (weights,gammas,pruning_masks,rand_maps,means,bn_betas,bn_gammas,bn_means,bn_inv_stds)

if __name__ == "__main__":

	last_conv_layer_id = 5
	last_layer_id = 7
	ls_layer_id = 5         # SKIP LAYER 5 HEADER GENERATION
	#ls_layer_id = -1		# DO NOT SKIP LAYER 5 LAYER GENERATION
		
	targetDirBin = "pretrained_network"	
	h5_file_name = targetDirBin+".h5"
	targetDirHLS = targetDirBin+"/hw"
	
	numRes = 2

	#topology of convolutional layers (only for config.h defines)
	ifm	   = [32, 30, 14, 12,  5]
	ofm	   = [30, 28, 12, 10,  3]   
	ifm_ch	= [ 3, 16, 16, 32, 32]
	ofm_ch	= [16, 16, 32, 32, 64]   
	filterDim = [ 3,  3,  3,  3,  3]

	WeightsPrecisions_integer =	   [1 , 1 , 1 , 1 , 1 , 1 , 1]
	WeightsPrecisions_fractional =	[0 , 0 , 0 , 0 , 0 , 0 , 0]

	InputPrecisions_integer =		 [1 , 1 , 1 , 1 , 1 , 1 , 1]
	InputPrecisions_fractional =	  [7 , 0 , 0 , 0 , 0 , 0 , 0]

	ActivationPrecisions_integer =	[1 , 1 , 1 , 1 , 1 , 1 , 16]
	ActivationPrecisions_fractional = [0 , 0 , 0 , 0 , 0 , 0 ,  0]

	classes = ['correct_mask', 'uncovered_chin', 'uncovered_mouth_nose', 'uncovered_nose']

	#configuration of PE and SIMD counts
	#BinCoP default configuration
	peCounts =   [4,  4,  4,  4,  1,  1, 1]
	simdCounts = [3, 16, 16, 32, 32, 16, 1]
	
	if not os.path.exists(targetDirBin):
		os.mkdir(targetDirBin)
	if not os.path.exists(targetDirHLS):
		os.mkdir(targetDirHLS)	

	# Loading the pretrained parameters
	weights,gammas,pruning_masks,rand_maps,means,bn_betas,bn_gammas,bn_means,bn_inv_stds = modelParametersList(h5_file_name, last_conv_layer_id, last_layer_id, ls_layer_id)

	# Format of thresholds, alphas, means_in, means_out
	numBits=28
	numIntBits=16

	config = "/**\n"
	config+= " * Finnthesizer Config-File Generation\n";
	config+= " *\n **/\n\n"
	config+= "#ifndef __LAYER_CONFIG_H_\n#define __LAYER_CONFIG_H_\n\n#define numResidual "+str(numRes)+"\n\n"

	for convl in range(0, last_conv_layer_id):
		if convl != (ls_layer_id-1):
			peCount = peCounts[convl]
			simdCount = simdCounts[convl]
			WPrecision_fractional = WeightsPrecisions_fractional[convl]
			APrecision_fractional = ActivationPrecisions_fractional[convl]
			IPrecision_fractional = InputPrecisions_fractional[convl]
			WPrecision_integer = WeightsPrecisions_integer[convl]
			APrecision_integer = ActivationPrecisions_integer[convl]
			IPrecision_integer = InputPrecisions_integer[convl]
			print("\nUsing peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, convl))
			
			usePopCount = False
			
			# generate weights and threshold
			w = weights[convl][0]
			w, thresholds = makeConvBNComplex(w, bn_betas[convl], bn_gammas[convl], bn_means[convl], bn_inv_stds[convl])

			# generate means_out and means_in
			if convl!=0:
				means_in = abs(means[convl-1][()])
			else:
				means_in = None
			next_means_b0 = abs(means[convl][0])
			means_out = np.transpose(np.array([makeNextLayerMeans(next_means_b0, ofm_ch[convl], bn_gammas[convl], bn_inv_stds[convl])]))

			# generate alphas
			alphas = []
			if convl!=0:
				for i in range(numRes):
					alphas.append(abs(gammas[convl] * means[convl-1][i]))
					print(means[convl-1][i])
			else:
				alphas.append(abs(gammas[convl]))
			
			# compute the padded width and height
			paddedH = padTo(w.shape[0], peCount)
			paddedW = padTo(w.shape[1], simdCount)

			# compute memory needed for weights and thresholds
			neededWMem = (paddedW * paddedH) // (simdCount * peCount)
			neededTMem = paddedH // peCount
			neededAMem = paddedH // peCount

			print("Layer %d: %d x %d" % (convl, paddedH, paddedW))
			print("WMem = %d TMem = %d, AMem = %d" % (neededWMem, neededTMem, neededAMem))
			print("IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, \
				WPrecision_integer,WPrecision_fractional, APrecision_integer, APrecision_fractional))
			
			m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, neededAMem, WPrecision_integer, APrecision_integer, IPrecision_integer, \
				WPrecision_fractional, APrecision_fractional, IPrecision_fractional, numBits=numBits, numIntBits=numIntBits, numRes=numRes)

			m.addMatrix(w,thresholds,alphas,means_out,means_in,paddedW,paddedH)

			config += (printConvDefines("L%d" % convl, filterDim[convl], ifm_ch[convl], ifm[convl], ofm_ch[convl], ofm[convl], simdCount, \
				peCount, neededWMem, neededTMem, WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 
			
			#generate HLS weight and threshold header file to initialize memory directly on bitstream generation		
			m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(convl) + ".h", str(convl))

	# process fully-connected layers
	for fcl in range(last_conv_layer_id,last_layer_id):
		peCount = peCounts[fcl]
		simdCount = simdCounts[fcl]
		WPrecision_fractional = WeightsPrecisions_fractional[fcl]
		APrecision_fractional = ActivationPrecisions_fractional[fcl]
		IPrecision_fractional = InputPrecisions_fractional[fcl]
		WPrecision_integer = WeightsPrecisions_integer[fcl]
		APrecision_integer = ActivationPrecisions_integer[fcl]
		IPrecision_integer = InputPrecisions_integer[fcl]
		print("\nUsing peCount = %d simdCount = %d for engine %d" % (peCount, simdCount, fcl))
		
		#usePopCount = True
		usePopCount = False

		# generate weights and threshold
		w = weights[fcl][0]
		outs = w.shape[1]
		if fcl!=last_layer_id-1:
			w, thresholds = makeFCBNComplex(w, bn_betas[fcl], bn_gammas[fcl], bn_means[fcl], bn_inv_stds[fcl])
			useThresholds = True
		else: # the last layer does not need threshold
			w, thresholds = makeFCBNComplex(w, bn_betas[fcl], bn_gammas[fcl], bn_means[fcl], bn_inv_stds[fcl])
			useThresholds = False

		# generate means_out and means_in
		if fcl!=last_layer_id-1:
			next_means_b0 = abs(means[fcl][0])
			means_out = np.transpose(np.array([makeNextLayerMeans(next_means_b0, outs, bn_gammas[fcl], bn_inv_stds[fcl])]))

		else:
			means_out = None
		means_in = abs(means[fcl-1][()])

		# generate alphas
		alphas = []
		for i in range(numRes):
			alphas.append(abs(gammas[fcl] * means[fcl-1][i]))

		# compute the padded width and height
		paddedH = padTo(w.shape[0], peCount)
		paddedW = padTo(w.shape[1], simdCount)
		
		# compute memory needed for weights and thresholds
		neededWMem = (paddedW * paddedH) // (simdCount * peCount)
		neededTMem = paddedH // peCount
		neededAMem = paddedH // peCount
		print("Layer %d: %d x %d" % (fcl, paddedH, paddedW))
		print("WMem = %d TMem = %d, AMem = %d" % (neededWMem, neededTMem, neededAMem))
		print("IPrecision = %d.%d WPrecision = %d.%d APrecision = %d.%d" % (IPrecision_integer, IPrecision_fractional, WPrecision_integer,\
			WPrecision_fractional, APrecision_integer, APrecision_fractional))

		m = BNNProcElemMem(peCount, simdCount, neededWMem, neededTMem, neededAMem, WPrecision_integer, APrecision_integer, IPrecision_integer, \
			WPrecision_fractional, APrecision_fractional, IPrecision_fractional, numBits=numBits, numIntBits=numIntBits, numRes=numRes)

		m.addMatrix(w,thresholds,alphas,means_out,means_in,paddedW,paddedH)

		config += (printFCDefines("L%d" % fcl, simdCount, peCount, neededWMem, neededTMem, paddedW, paddedH, \
			WPrecision_integer, APrecision_integer, WPrecision_fractional, APrecision_fractional)) + "\n" 
		
		#generate HLS weight and threshold header file to initialize memory directly on bitstream generation
		m.createHLSInitFiles(targetDirHLS + "/memdata-" + str(fcl) + ".h", str(fcl), useThresholds)

	config+="#endif //__LAYER_CONFIG_H_\n"
	configFile = open(targetDirHLS+"/config.h", "w")
	configFile.write(config)
	configFile.close()

	with open(targetDirBin+"/classes.txt", "w") as f:
		f.write("\n".join(classes))
