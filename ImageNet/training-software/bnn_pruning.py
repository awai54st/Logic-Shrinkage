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

		if 'binary' in key:	
			bl_w1 = bl["model_weights"][key][key]["Variable_1:0"]
			bl_pruning_mask = bl["model_weights"][key][key]["pruning_mask:0"]
			#bl_gamma = bl["model_weights"][key][key]["Variable:0"]
			zero_fill = np.zeros(np.shape(np.array(bl_w1)))
			pret_w1 = pretrained["model_weights"][key][key]["Variable_1:0"]
			pret_pruning_mask = pretrained["model_weights"][key][key]["pruning_mask:0"]
			#p_gamma = pretrained["model_weights"][key][key]["Variable:0"]
			
			pret_w1[...] = np.array(bl_w1)
			#p_gamma[...] = np.array(bl_gamma)


			if normalisation=="l1":
				norm=abs(np.array(bl_w1))
			elif normalisation=="l2":
				norm=np.array(bl_w1)**2
				norm=np.sqrt(norm)
			if 'conv' in key:
				norm=np.reshape(norm, [-1,np.shape(norm)[3]])
			
#			weight = np.array(bl_w1)
#			if "conv" in key:
#				Tsize_RC = np.shape(weight)[0]
#				Tsize_M = np.shape(weight)[2]/TM
#				Tsize_N = np.shape(weight)[3]/TN
#				one_tile = np.zeros([Tsize_RC,Tsize_RC,Tsize_M,Tsize_N])
#				# set up pruning_mask
#				#mean=np.mean(abs(weight),axis=3)
#				norm=one_tile
#				if normalisation=="l1":
#					for n in range(TN):
#						for m in range(TM):
#							norm = norm + weight[0:Tsize_RC,0:Tsize_RC,(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
#					norm = norm / (TM*TN)
#				elif normalisation=="l2":
#					for n in range(TN):
#						for m in range(TM):
#							norm = norm + weight[0:Tsize_RC,0:Tsize_RC,(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
#					norm = norm / (TM*TN)
#					norm = np.sqrt(norm)
#				norm=np.reshape(norm, [-1,np.shape(norm)[3]])
#			elif "dense" in key:
#				Tsize_M = np.shape(weight)[0]/TM
#				Tsize_N = np.shape(weight)[1]/TN
#				one_tile = np.zeros([Tsize_M,Tsize_N])
#				# set up pruning_mask
#				#mean=np.mean(abs(weight),axis=3)
#				norm=one_tile
#				if normalisation=="l1":
#					for n in range(TN):
#						for m in range(TM):
#							norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]
#					norm = norm / (TM*TN)
#				elif normalisation=="l2":
#					for n in range(TN):
#						for m in range(TM):
#								norm = norm + weight[(m*Tsize_M):((m+1)*Tsize_M),(n*Tsize_N):((n+1)*Tsize_N)]**2
#					norm = norm / (TM*TN)
#					norm = np.sqrt(norm)
			
			if pruning_percentage[key] < 0:
				pruning_mask = np.ones_like(norm)
				
			else:
                #iterative method
				target=pruning_percentage[key]
				#deviation=0.010
				#min_val=target-deviation
				#max_val=target+deviation
				#initial=pruning_threshold[key]
				#step=initial/2
				#val=initial
				#pruning_mask = np.greater(norm, val)
				#out=((np.count_nonzero(pruning_mask==0))*1.0/pruning_mask.size)
				#while (out>max_val)or(out<min_val):
				#	d=out
				#	val=val+step
				#	pruning_mask = np.greater(norm, val)
				#	pruning_ratio=((np.count_nonzero(pruning_mask==0))*1.0/pruning_mask.size)
				#	out=pruning_ratio
				#	j=1
				#	gradient=(out-d)/(j*step)
				#	dis=target-out
				#	step=dis/gradient

				norm_shape = np.shape(norm)
				mag_sort_idx = np.argsort(norm.flatten())
				mag_sort_idx = np.argsort(mag_sort_idx) # argsort twice to get the rank of the array
				pruning_mask_flat = np.greater(mag_sort_idx, target * np.prod(norm_shape)) # mask the target percentile
				#pruning_mask_flat = np.greater(mag_sort_idx, 0.5 * np.prod(norm_shape)) # mask the target percentile
				pruning_mask = np.reshape(pruning_mask_flat, norm_shape)
                    
                    
			pret_pruning_mask[...] = np.array(pruning_mask,dtype=float)
			
			pruning_ratio=((np.count_nonzero(pruning_mask==0))*1.0/pruning_mask.size)
			
		#if b'residual_sign' in key:	
		#	bl_means = bl["model_weights"][key][key]["means:0"]
		#	pret_means = pretrained["model_weights"][key][key]["means:0"]
		#	pret_means[...] = np.array(bl_means)
	
	pretrained.close()
	
	copyfile(output_path + "/pretrained_pruned.h5", output_path + "/2_residuals.h5") 
