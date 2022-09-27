import h5py
import numpy as np
import sys

from shutil import copyfile

def bnn_pruning(get_model, pruning_percentage, output_path='models'):

	copyfile(output_path + "/2_residuals.h5", output_path + "/bnn.h5") # create pretrained.h5 using datastructure from dummy.h5
	copyfile(output_path + "/2_residuals.h5", output_path + "/pretrained_pruned.h5") # create pretrained.h5 using datastructure from dummy.h5
	bl = h5py.File(output_path + "/2_residuals.h5", 'r')
	pretrained = h5py.File(output_path + "/pretrained_pruned.h5", 'r+')
	
	normalisation="l2"
	
	for key in bl['model_weights'].attrs['layer_names']:

		if b'binary' in key:	
			bl_w1 = bl["model_weights"][key][key]["Variable_1:0"]
			bl_pruning_mask = bl["model_weights"][key][key]["pruning_mask:0"]
			bl_gamma = bl["model_weights"][key][key]["Variable:0"]
			zero_fill = np.zeros(np.shape(np.array(bl_w1)))
			pret_w1 = pretrained["model_weights"][key][key]["Variable_1:0"]
			pret_pruning_mask = pretrained["model_weights"][key][key]["pruning_mask:0"]
			p_gamma = pretrained["model_weights"][key][key]["Variable:0"]
			
			pret_w1[...] = np.array(bl_w1)
			p_gamma[...] = np.array(bl_gamma)


			if normalisation=="l1":
				norm=abs(np.array(bl_w1))
			elif normalisation=="l2":
				norm=np.array(bl_w1)**2
				norm=np.sqrt(norm)
			if b'conv' in key:
				norm=np.reshape(norm, [-1,np.shape(norm)[3]])
			
			if pruning_percentage[key] < 0:
				pruning_mask = np.ones_like(norm)
				
			else:
                #iterative method
				target=pruning_percentage[key]

				norm_shape = np.shape(norm)
				mag_sort_idx = np.argsort(norm.flatten())
				mag_sort_idx = np.argsort(mag_sort_idx) # argsort twice to get the rank of the array
				pruning_mask_flat = np.greater(mag_sort_idx, target * np.prod(norm_shape)) # mask the target percentile
				pruning_mask = np.reshape(pruning_mask_flat, norm_shape)
                    
                    
			pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
			
			pruning_ratio=((np.count_nonzero(pruning_mask==0))*1.0/pruning_mask.size)
	
	pretrained.close()
	
	copyfile(output_path + "/pretrained_pruned.h5", output_path + "/2_residuals.h5") 
