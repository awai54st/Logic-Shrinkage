import h5py
import numpy as np
import sys
from tensorflow import keras
import tensorflow as tf

from dataset import get_dataset

from shutil import copyfile

#Kernel generation functions using Numpy
def genkernel (K,i):
    eye1=np.identity(2**i)
    eye2=np.identity(2**(K-1-i))
    
    mat=np.kron(np.ones([2,2]),eye2)
    
    mat=0.5*np.kron(eye1,mat)
    return mat

def genkernel_mat (K,mask,num_ops):
    out=np.kron(np.ones([num_ops,1,1]),np.identity(2**K))
    for i in range(K):
        c=np.where(mask[i],genkernel(K,i),np.identity(2**K))
        out=np.matmul(out,c)
    return out

def logic_shrinkage(get_model, K, activation_pruning_percentage, output_path, custom_rand_seed):

    # TODO: temporary declaration of ls layer list here
    # If CIFAR-10 or SVHN
    first_layer = b'binary_conv_1'
    ls_layers = [b'binary_conv_6']
    # If MNIST
    #first_layer = b'binary_dense_1'
    #ls_layers = [b'binary_dense_2',b'binary_dense_3',b'binary_dense_4']

    shrinkage_strategy = "magnitude"

    
    copyfile(output_path + "/2_residuals.h5", output_path + "/preshrinked.h5")
    shrinked= h5py.File(output_path + "/2_residuals.h5", 'r+')
    preshrinked= h5py.File(output_path + "/preshrinked.h5", 'r+')
    
    for key in shrinked['model_weights'].attrs['layer_names']:
        for ls_layer in ls_layers:

            if ls_layer in key:

                #Load shrinkage map
                if K > 4: # Split shrinkage_map into 8 parts to avoid the 2GB tensor limit
                    shrinkage_map_p0=shrinked['model_weights'][key][key]['shrinkage_map_p0:0']
                    shrinkage_map_p1=shrinked['model_weights'][key][key]['shrinkage_map_p1:0']
                    shrinkage_map_p2=shrinked['model_weights'][key][key]['shrinkage_map_p2:0']
                    shrinkage_map_p3=shrinked['model_weights'][key][key]['shrinkage_map_p3:0']
                    shrinkage_map_p4=shrinked['model_weights'][key][key]['shrinkage_map_p4:0']
                    shrinkage_map_p5=shrinked['model_weights'][key][key]['shrinkage_map_p5:0']
                    shrinkage_map_p6=shrinked['model_weights'][key][key]['shrinkage_map_p6:0']
                    shrinkage_map_p7=shrinked['model_weights'][key][key]['shrinkage_map_p7:0']
                else:
                    shrinkage_map=shrinked['model_weights'][key][key]['shrinkage_map:0']
                pruning_mask=shrinked['model_weights'][key][key]['pruning_mask:0']
                                
                #---Start of shrinkage strategy---
                
                
                #Load pre-shrinkage parameters
                ps_c_param = shrinked['model_weights'][key][key]['Variable_1:0']

                ps_act_1 = ps_c_param[0:(2**K)]
                ps_act_2 = ps_c_param[(2**K):((2**K)*2)]
                
                # mask is the selection matrix from which the map is created
                num_ops = np.array(ps_c_param[0]).flatten().size
                mask_shape = [K, num_ops]
                mask=np.zeros(mask_shape)

                if shrinkage_strategy == "magnitude":

                    # magnitude-based activation pruning, with a kernel that evaluates each activation's relative importance
                    weight=np.zeros([K,num_ops])
                    for i in range(K):
                        for b in range(2**i):
                            for c in range(2**(K-i-1)):
                                p = (b*2**(K-i)+c)
                                q = p+2**(K-i-1)
                                
                                weight[i]+=np.abs(ps_act_1[p].flatten()-ps_act_1[q].flatten())
                                weight[i]+=np.abs(ps_act_2[p].flatten()-ps_act_2[q].flatten())

                    ## SAME N PER LUT    

                    #for i in range(K):
                    #    weight[i] = np.where(np.array(pruning_mask).flatten(), weight[i], -1.0)

                    #weight_sorted = np.argsort(weight, axis=0)
                    #weight_sorted = np.argsort(weight_sorted, axis=0)
           
                    #for i in range(K):
                    #    mask[i] = np.where(weight_sorted[i] < num_act_pruned, 1, 0) # flag the least salient connection

                    # DIFFERENT N PER LUT    
                    # Sort the saliencies per K

                    for i in range(K):
                        weight[i] = np.where(np.array(pruning_mask).flatten(), weight[i], -1.0)

                    # New tweak: having scores that take into account both itself and the sum in LUT
                    #weight += np.sum(weight, axis=0)
                    ######

                    #k_count = np.sum(np.logical_not(np.squeeze(mask)), axis=0)
                    #for i in range(K):
                    #    #weight[i] = np.divide(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)
                    #    weight[i] = np.multiply(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)

                    saliency_scores_sorted = np.argsort(weight, axis=None)
                    saliency_scores_sorted = np.argsort(saliency_scores_sorted, axis=None).reshape((K, -1))

                    lut_pruned_count = np.sum(np.logical_not(pruning_mask))
                    mask = np.where(saliency_scores_sorted < K * lut_pruned_count + (np.max(saliency_scores_sorted)-K*lut_pruned_count) * (activation_pruning_percentage), 1, 0)
                    #mask = np.where(saliency_scores_sorted < K * lut_pruned_count + (np.max(saliency_scores_sorted)-K*lut_pruned_count) * (activation_pruning_percentage*0.33), 1, 0)

                    ## Special version: penalise weights wrt K; multiple iterations of pruning

                    #for i in range(K):
                    #    weight[i] = np.where(np.array(pruning_mask).flatten(), weight[i], -1.0)

                    #k_count = np.sum(np.logical_not(np.squeeze(mask)), axis=0)
                    #for i in range(K):
                    #    #weight[i] = np.divide(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)
                    #    weight[i] = np.multiply(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)

                    #saliency_scores_sorted = np.argsort(weight, axis=None)
                    #saliency_scores_sorted = np.argsort(saliency_scores_sorted, axis=None).reshape((K, -1))
                    #mask = np.where(saliency_scores_sorted < K * lut_pruned_count + (np.max(saliency_scores_sorted)-K*lut_pruned_count) * (activation_pruning_percentage*0.67), 1, 0)

                    #for i in range(K):
                    #    weight[i] = np.where(np.array(pruning_mask).flatten(), weight[i], -1.0)

                    #k_count = np.sum(np.logical_not(np.squeeze(mask)), axis=0)

                    #for i in range(K):
                    #    #weight[i] = np.where(k_count == 0.0, 0.0, weight[i] / k_count)
                    #    #weight[i] = np.divide(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)
                    #    weight[i] = np.multiply(weight[i], k_count, out=np.zeros_like(weight[i]), where=k_count!=0)

                    #saliency_scores_sorted = np.argsort(weight, axis=None)
                    #saliency_scores_sorted = np.argsort(saliency_scores_sorted, axis=None).reshape((K, -1))
                    #mask = np.where(saliency_scores_sorted < K * lut_pruned_count + (np.max(saliency_scores_sorted)-K*lut_pruned_count) * activation_pruning_percentage, 1, 0)

                    ## Revised special version: penalise weights wrt K; multiple iterations of pruning, for loop through each sample
                    #lut_pruned_count = np.sum(np.logical_not(pruning_mask))
                    #mask = np.zeros_like(weight)
                    #for i in range(K):
                    #    weight[i] = np.where(np.array(pruning_mask).flatten(), weight[i], -1.0)

                    #remaining_acts = weight.size - K*lut_pruned_count

                    #for pruned_acts_incr in range(int(remaining_acts * activation_pruning_percentage)):

                    #    k_count = np.sum(np.logical_not(np.squeeze(mask)), axis=0)

                    #    for i in range(K):
                    #        weight[i] = np.divide(weight[i], k_count, out=np.zeros_like(weight[i])-1, where=k_count!=0)
                    #    saliency_scores_sorted = np.argsort(weight, axis=None)
                    #    saliency_scores_sorted = np.argsort(saliency_scores_sorted, axis=None).reshape((K, -1))
                    #    mask = np.where(saliency_scores_sorted < K * lut_pruned_count + pruned_acts_incr, 1, mask)
                    #    weight = np.where(mask, -1.0, weight)
                   

                #---End of shrinkage strategy
                mask=np.reshape(mask,[K,num_ops,1,1])
                #shrinkage_map[...]=genkernel_mat(K,mask=mask,num_ops=num_ops)
                shrinkage_map_unpruned=genkernel_mat(K,mask=mask,num_ops=num_ops)
                if K > 4: # Split shrinkage_map into 8 parts to avoid the 2GB tensor limit
                    shrinkage_map_unpruned_splitted = np.split(shrinkage_map_unpruned, 8, axis=-1)
                    shrinkage_map_p0[...]=shrinkage_map_unpruned_splitted[0]
                    shrinkage_map_p1[...]=shrinkage_map_unpruned_splitted[1]
                    shrinkage_map_p2[...]=shrinkage_map_unpruned_splitted[2]
                    shrinkage_map_p3[...]=shrinkage_map_unpruned_splitted[3]
                    shrinkage_map_p4[...]=shrinkage_map_unpruned_splitted[4]
                    shrinkage_map_p5[...]=shrinkage_map_unpruned_splitted[5]
                    shrinkage_map_p6[...]=shrinkage_map_unpruned_splitted[6]
                    shrinkage_map_p7[...]=shrinkage_map_unpruned_splitted[7]
                else:
                    shrinkage_map[...]=shrinkage_map_unpruned

                # Check if all act connections in a LUT have been removed.
                # If so, prune the LUT.
                bool_all_elems_equal = np.zeros((num_ops,), dtype=bool)
                for op_idx in range(num_ops):
                    bool_all_elems_equal[op_idx] = np.all(shrinkage_map_unpruned[op_idx] == shrinkage_map_unpruned[op_idx][0][0])
                #print(bool_all_elems_equal)
                #print(np.sum(bool_all_elems_equal))

                #act_pruning_mask = np.logical_not(bool_all_elems_equal).reshape(np.shape(pruning_mask))
                act_pruning_mask = np.where(np.sum(np.reshape(mask, [K,-1]), axis=0) == K, 0.0, 1.0).reshape(np.shape(pruning_mask))
                pruning_mask[...] = np.logical_and(pruning_mask, act_pruning_mask)

                # Print post-pruning parameter count and lut count
                k_count = np.sum(np.logical_not(np.squeeze(mask)), axis=0)
                # Cases where lut has been pruned
                param_count = np.where(np.array(pruning_mask).flatten(), (2 ** k_count) * 2, 0)
                print("Layer: ", key)
                print("Parameter count: ", np.sum(param_count))
                print("LUT count: ", np.sum(pruning_mask))
                print("Act count: ", np.sum(k_count))
                #print(k_count.tolist())
                #print(param_count.tolist())


