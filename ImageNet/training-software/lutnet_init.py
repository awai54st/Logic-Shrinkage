import h5py
import numpy as np
import sys

from shutil import copyfile

def lutnet_init(K, get_model, output_path, custom_rand_seed):

	# If CIFAR-10 or SVHN
	#target_layers = [b'binary_conv_6']
	# If MNIST
	#target_layers = [b'binary_dense_2',b'binary_dense_3',b'binary_dense_4',b'binary_dense_5']
	# If IMAGENET
	target_layers = ['binary_conv_12']

	#np.random.seed(custom_rand_seed)

	copyfile(output_path + "/dummy.h5", output_path + "/bnn_pruned.h5") # create pretrained.h5 using datastructure from dummy.h5
	copyfile(output_path + "/dummy.h5", output_path + "/pretrained_bin.h5") # create pretrained.h5 using datastructure from dummy.h5
	
	bl = h5py.File(output_path + "/2_residuals.h5", 'r')
	pretrained = h5py.File(output_path + "/pretrained_bin.h5", 'r+')

	for key in bl['model_weights'].attrs['layer_names']:

		#if b'binary_conv_6' in key:
		#for target_layer in target_layers:
		#	if target_layer in key:

		if any(target_layer in key for target_layer in target_layers):
	
			bl_w1 = bl["model_weights"][key][key]["Variable_1:0"]
			bl_rand_map_0 = bl["model_weights"][key][key]["rand_map_0:0"]
			bl_pruning_mask = bl["model_weights"][key][key]["pruning_mask:0"]
			#bl_gamma = bl["model_weights"][key][key]["Variable:0"]
			bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]
			if K > 1:
				pret_rand_map_0 = pretrained["model_weights"][key][key]["rand_map_0:0"]
			if K > 2:
				pret_rand_map_1 = pretrained["model_weights"][key][key]["rand_map_1:0"]
			if K > 3:
				pret_rand_map_2 = pretrained["model_weights"][key][key]["rand_map_2:0"]
			if K > 4:
				pret_rand_map_3 = pretrained["model_weights"][key][key]["rand_map_3:0"]
			if K > 5:
				pret_rand_map_4 = pretrained["model_weights"][key][key]["rand_map_4:0"]

			pret_pruning_mask = pretrained["model_weights"][key][key]["pruning_mask:0"]
			#p_gamma = pretrained["model_weights"][key][key]["Variable:0"]
			pret_means = pretrained["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

			pret_c_param = pretrained["model_weights"][key][key]["Variable_1:0"]
			
			weight_shape = np.shape(bl_w1)
			
			if 'binary_conv' in key:
				# randomisation and pruning recovery
				bl_w1_unroll = np.reshape(np.array(bl_w1), (-1,weight_shape[3]))
				bl_w1 = np.array(bl_w1)
				
				if K > 1:
					rand_map_0 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
					np.random.shuffle(rand_map_0)
				if K > 2:
					rand_map_1 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
					np.random.shuffle(rand_map_1)
				if K > 3:
					rand_map_2 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
					np.random.shuffle(rand_map_2)
				if K > 4:
					rand_map_3 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
					np.random.shuffle(rand_map_3)
				if K > 5:
					rand_map_4 = np.arange(weight_shape[0]*weight_shape[1]*weight_shape[2])
					np.random.shuffle(rand_map_4)

			elif 'binary_dense' in key:
				# randomisation and pruning recovery
				bl_w1_unroll = np.array(bl_w1)
				bl_w1 = np.array(bl_w1)
				
				if K > 1:
					rand_map_0 = np.arange(weight_shape[0])
					np.random.shuffle(rand_map_0)
				if K > 2:
					rand_map_1 = np.arange(weight_shape[0])
					np.random.shuffle(rand_map_1)
				if K > 3:
					rand_map_2 = np.arange(weight_shape[0])
					np.random.shuffle(rand_map_2)
				if K > 4:
					rand_map_3 = np.arange(weight_shape[0])
					np.random.shuffle(rand_map_3)
				if K > 5:
					rand_map_4 = np.arange(weight_shape[0])
					np.random.shuffle(rand_map_4)

			c_param = [None] * ((2**K)*2)
			pruning_mask = np.array(bl_pruning_mask).astype(bool)
			
			# weights for extra input 1
			if K > 1:
				init_mask = np.logical_not(pruning_mask[rand_map_0])
				pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_0)]
				pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
				init_mask = np.reshape(init_mask, weight_shape)
	
				bl_w1_rand = bl_w1_unroll[rand_map_0]
				bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)

				for c_idx in range((2**K)*2):
					c_param[c_idx] = ((-1)**(c_idx/((2**K)/2))) * bl_w1
	
				for c_idx in range((2**K)*2):
					c_param[c_idx][init_mask] = c_param[c_idx][init_mask] + ((-1)**(c_idx/((2**K)/4))) * bl_w1_rand[init_mask]
					
			# weights for extra input 2
			if K > 2:
				init_mask = np.logical_not(pruning_mask[rand_map_1])
				pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_1)]
				pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
				init_mask = np.reshape(init_mask, weight_shape)
				
				bl_w1_rand = bl_w1_unroll[rand_map_1]
				bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)
		
				for c_idx in range((2**K)*2):
					c_param[c_idx][init_mask] = c_param[c_idx][init_mask] + ((-1)**(c_idx/((2**K)/8))) * bl_w1_rand[init_mask]
			
			# weights for extra input 3
			if K > 3:
				init_mask = np.logical_not(pruning_mask[rand_map_2])
				pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_2)]
				pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
				init_mask = np.reshape(init_mask, weight_shape)
				
				bl_w1_rand = bl_w1_unroll[rand_map_2]
				bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)
			
				for c_idx in range((2**K)*2):
					c_param[c_idx][init_mask] = c_param[c_idx][init_mask] + ((-1)**(c_idx/((2**K)/16))) * bl_w1_rand[init_mask]
				
			# weights for extra input 4
			if K > 4:
				init_mask = np.logical_not(pruning_mask[rand_map_3])
				pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_3)]
				pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
				init_mask = np.reshape(init_mask, weight_shape)
				
				bl_w1_rand = bl_w1_unroll[rand_map_3]
				bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)
			
				for c_idx in range((2**K)*2):
					c_param[c_idx][init_mask] = c_param[c_idx][init_mask] + ((-1)**(c_idx/((2**K)/32))) * bl_w1_rand[init_mask]
		
			# weights for extra input 5
			if K > 5:
				init_mask = np.logical_not(pruning_mask[rand_map_4])
				pruning_mask_recover = np.logical_and(pruning_mask, init_mask)[np.argsort(rand_map_4)]
				pruning_mask = np.logical_or(pruning_mask, pruning_mask_recover)
				init_mask = np.reshape(init_mask, weight_shape)
				
				bl_w1_rand = bl_w1_unroll[rand_map_4]
				bl_w1_rand = np.reshape(bl_w1_rand, weight_shape)
			
				for c_idx in range((2**K)*2):
					c_param[c_idx][init_mask] = c_param[c_idx][init_mask] + ((-1)**(c_idx/((2**K)/64))) * bl_w1_rand[init_mask]

			pret_c_param[...] = np.array(np.stack(c_param, axis=0), dtype=float)
			
			if K > 1:
				pret_rand_map_0[...] = np.reshape(rand_map_0, (-1,1)).astype(float)
			if K > 2:
				pret_rand_map_1[...] = np.reshape(rand_map_1, (-1,1)).astype(float)
			if K > 3:
				pret_rand_map_2[...] = np.reshape(rand_map_2, (-1,1)).astype(float)
			if K > 4:
				pret_rand_map_3[...] = np.reshape(rand_map_3, (-1,1)).astype(float)
			if K > 5:
				pret_rand_map_4[...] = np.reshape(rand_map_4, (-1,1)).astype(float)

			p_gamma[...] = np.array(bl_gamma)
			#pret_means[...] = np.array(bl_means)
			pret_pruning_mask[...] = np.array(bl_pruning_mask)
			
			print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

		elif 'binary_' in key:

			bl_w1 = bl["model_weights"][key][key]["Variable_1:0"]
			bl_pruning_mask = bl["model_weights"][key][key]["pruning_mask:0"]
			#bl_gamma = bl["model_weights"][key][key]["Variable:0"]
			zero_fill = np.zeros(np.shape(np.array(bl_w1)))
			pret_w1 = pretrained["model_weights"][key][key]["Variable_1:0"]
			pret_pruning_mask = pretrained["model_weights"][key][key]["pruning_mask:0"]
			#p_gamma = pretrained["model_weights"][key][key]["Variable:0"]
			
			pret_w1[...] = np.array(bl_w1)
			#p_gamma[...] = np.array(bl_gamma)
			pret_pruning_mask[...] = np.array(bl_pruning_mask)
			
			print(np.sum(np.array(bl_pruning_mask)), np.prod(np.shape(np.array(bl_pruning_mask))))

		elif 'residual_sign' in key:
			bl_means = bl["model_weights"][key][key]["means:0"]
			pret_means = pretrained["model_weights"][key][key]["means:0"]
			pret_means[...] = np.array(bl_means)

		elif 'batch_normalization' in key:
			bl_beta = bl["model_weights"][key][key]["beta:0"]
			bl_gamma = bl["model_weights"][key][key]["gamma:0"]
			bl_moving_mean = bl["model_weights"][key][key]["moving_mean:0"]
			bl_moving_variance = bl["model_weights"][key][key]["moving_variance:0"]
			p_beta = pretrained["model_weights"][key][key]["beta:0"]
			p_gamma = pretrained["model_weights"][key][key]["gamma:0"]
			p_moving_mean = pretrained["model_weights"][key][key]["moving_mean:0"]
			p_moving_variance = pretrained["model_weights"][key][key]["moving_variance:0"]
			
			p_beta[...] = np.array(bl_beta)
			p_gamma[...] = np.array(bl_gamma)
			p_moving_mean[...] = np.array(bl_moving_mean)
			p_moving_variance[...] = np.array(bl_moving_variance)

	pretrained.close()
	
	copyfile(output_path + "/pretrained_bin.h5", output_path + "/2_residuals.h5")
