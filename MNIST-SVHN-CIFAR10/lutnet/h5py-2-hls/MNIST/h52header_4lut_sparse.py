import h5py
import numpy as np

def SignNumpy(x):
  return np.greater(x,0)

# convert a fully connected binarized layer plus batch normalization into 
# the simplified form (binary weight and positive threshold)
# note that the neurons are assumed to be in the columns of the weight
# matrix
def makeBNComplex(after_bn_thres, fanin, beta, gamma, mean, invstd, use_rowmajor=False, usePopCount=True):
  outs = fanin.shape[0]
  print ("Extracting FCBN complex, outs = %d" % (outs))
  # we'll fill in the binarized weights and thresholds iteratively
#  w_bin = range(ins*outs)
  thresholds = list(range(outs))
  for neuron in range(outs):
    # compute a preliminary threshold from the batchnorm parameters
    thres = mean[neuron] + ((after_bn_thres - beta[neuron]) / (abs(gamma[neuron]*invstd[neuron])+1e-4))
    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
#    if gamma[neuron]*invstd[neuron] < 0:
#        need_flip = 1
#        thres = -thres
#    if thres > 32767:
#        thres = 32767
#    if thres < -32768:
#        thres = -32768
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        #thresholds[neuron] = int((fanin[neuron] + thres) / 2)
        thresholds[neuron] = (fanin[neuron] + thres) / 2
    else:
        thresholds[neuron] = thres
#    # binarize the synapses
#    for synapse in range(ins):
#      # note how we change from col major to row major if requested
#      dest_ind = neuron*ins+synapse if use_rowmajor else synapse*outs+neuron
#      if need_flip:
#        w_bin[dest_ind] = binarize(-weights[synapse][neuron])
#      else:
#        w_bin[dest_ind] = binarize(weights[synapse][neuron])
#  # reshape the output as desired
#  if use_rowmajor:
#    w_bin = np.asarray(w_bin).reshape((outs, ins))
#  else:
#    w_bin = np.asarray(w_bin).reshape((ins, outs))
    
#return (w_bin, thresholds)
  return thresholds


# binarize and pack convolutional layer weights into a matrix and compute
# thresholds from the conv bias and batchnorm parameters
def makeConvBNComplex(fanin, beta, gamma, mean, invstd, interleaveChannels=False, usePopCount=True):
  numOut = fanin.shape[0]
  print ("Extracting conv-BN complex, OFM=%d" % (numOut))
  # the fanin is used to ensure positive-only threshold
#  w_bin = range(numOut * numIn * k * k)
  # one threshold per output channel
  thresholds = range(numOut)
#  dest_ind = 0
  # we'll fill in the binarized weights and thresholds iteratively
  for neuron in range(numOut):
    # compute a preliminary threshold from the batchnorm parameters,
    # subtracting the conv bias from the batchnorm mean
    thres = mean[neuron] - (beta[neuron] / (gamma[neuron]*invstd[neuron]))
#    need_flip = 0
    # ensure all neurons activate on the "positive" side, so we can use
    # greater-than-threshold activation
    if gamma[neuron]*invstd[neuron] < 0:
#        need_flip = 1
        thres = -thres
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds[neuron] = int((fanin[neuron] + thres) / 2)
    else:
        thresholds[neuron] = thres
#    # go through each weight of each convolutional kernel
#    if interleaveChannels:
#      for ky in range(k):
#        for kx in range(k):
#          for ifm in range(numIn):
#            f = -1 if need_flip else +1
#            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
#            dest_ind += 1
#    else:
#      for ifm in range(numIn):
#        for ky in range(k):
#          for kx in range(k):
#            f = -1 if need_flip else +1
#            w_bin[dest_ind] = binarize(f*weights[neuron][ifm][ky][kx])
#            dest_ind += 1
#          
#  # reshape the output as desired
#  w_bin = np.asarray(w_bin).reshape((numOut, fanin))
#  return (w_bin, thresholds)
  return thresholds

if __name__ == "__main__":

    print("Loading the pretrained parameters...")

    logic_shrinkage = True

    bl = h5py.File("pretrained_network_4lut.h5", 'r')
    #bl = h5py.File("dummy.h5", 'r')
    
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
    

    # dense layer 1
    
    bl_w1 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"])
    bl_rand_map = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_0:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_1"]["batch_normalization_1"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_1"]["residual_sign_1"]["means:0"]

    ##Pruning
    #bl_w1 = bl_w1 * np.reshape(bl_pruning_mask, (bl_w1.shape))
    
    w_lut = [bl_w1]
    #weights = [weights, w_lut]
    weights = [w_lut]
    #gammas = [gammas, bl_gamma]
    gammas=[bl_gamma]
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks=[bl_pruning_mask]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps=[bl_rand_map]
    #means = [means, bl_means]
    means=[bl_means]
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas=[bl_bn_beta]
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas=[bl_bn_gamma]
    #bn_means = [bn_means, bl_bn_mean]
    bn_means=[bl_bn_mean]
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds=[bl_bn_inv_std]
    
    # dense layer 2
    bl_c_param = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"])

    bl_shrinkage_map = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["shrinkage_map:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_2"]["batch_normalization_2"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_2"]["residual_sign_2"]["means:0"]

    # Apply logic shrinkage
    c_mat1=bl_c_param[0 : 2**4]
    c_mat2=bl_c_param[2**4 : (2**4)*2]
    half_c_mat_shape = np.shape(c_mat1)

    # Whether to apply shrinkage
    if logic_shrinkage:

        c_mat1=np.reshape(c_mat1,(1,2**4,-1))
        c_mat2=np.reshape(c_mat2,(1,2**4,-1))

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1=np.matmul(c_mat1,bl_shrinkage_map)
        #c_mat1=chunking_dot(c_mat1,bl_shrinkage_map)
        c_mat2=np.matmul(c_mat2,bl_shrinkage_map)

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1 = np.reshape(c_mat1, half_c_mat_shape)
        c_mat2 = np.reshape(c_mat2, half_c_mat_shape)

    c_mat = np.concatenate((c_mat1, c_mat2), axis=0)

    w_lut = c_mat * bl_pruning_mask
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

    # dense layer 3
    
    bl_c_param = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"])
 
    bl_shrinkage_map = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["shrinkage_map:0"])
   
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_3"]["batch_normalization_3"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_3"]["residual_sign_3"]["means:0"]

    # Apply logic shrinkage
    c_mat1=bl_c_param[0 : 2**4]
    c_mat2=bl_c_param[2**4 : (2**4)*2]
    half_c_mat_shape = np.shape(c_mat1)

    # Whether to apply shrinkage
    if logic_shrinkage:

        c_mat1=np.reshape(c_mat1,(1,2**4,-1))
        c_mat2=np.reshape(c_mat2,(1,2**4,-1))

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1=np.matmul(c_mat1,bl_shrinkage_map)
        #c_mat1=chunking_dot(c_mat1,bl_shrinkage_map)
        c_mat2=np.matmul(c_mat2,bl_shrinkage_map)

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1 = np.reshape(c_mat1, half_c_mat_shape)
        c_mat2 = np.reshape(c_mat2, half_c_mat_shape)

    c_mat = np.concatenate((c_mat1, c_mat2), axis=0)

    w_lut = c_mat * bl_pruning_mask
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

    # dense layer 4
     
    bl_c_param = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["Variable:0"])
 
    bl_shrinkage_map = np.array(bl["model_weights"]["binary_dense_4"]["binary_dense_4"]["shrinkage_map:0"])
   
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_4"]["batch_normalization_4"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]

    # Apply logic shrinkage
    c_mat1=bl_c_param[0 : 2**4]
    c_mat2=bl_c_param[2**4 : (2**4)*2]
    half_c_mat_shape = np.shape(c_mat1)

    # Whether to apply shrinkage
    if logic_shrinkage:

        c_mat1=np.reshape(c_mat1,(1,2**4,-1))
        c_mat2=np.reshape(c_mat2,(1,2**4,-1))

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1=np.matmul(c_mat1,bl_shrinkage_map)
        #c_mat1=chunking_dot(c_mat1,bl_shrinkage_map)
        c_mat2=np.matmul(c_mat2,bl_shrinkage_map)

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1 = np.reshape(c_mat1, half_c_mat_shape)
        c_mat2 = np.reshape(c_mat2, half_c_mat_shape)

    c_mat = np.concatenate((c_mat1, c_mat2), axis=0)

    w_lut = c_mat * bl_pruning_mask
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

    # dense layer 5
     
    bl_c_param = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)
    bl_gamma = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["Variable:0"])
 
    bl_shrinkage_map = np.array(bl["model_weights"]["binary_dense_5"]["binary_dense_5"]["shrinkage_map:0"])
   
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)

    # Apply logic shrinkage
    c_mat1=bl_c_param[0 : 2**4]
    c_mat2=bl_c_param[2**4 : (2**4)*2]
    half_c_mat_shape = np.shape(c_mat1)

    # Whether to apply shrinkage
    if logic_shrinkage:

        c_mat1=np.reshape(c_mat1,(1,2**4,-1))
        c_mat2=np.reshape(c_mat2,(1,2**4,-1))

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1=np.matmul(c_mat1,bl_shrinkage_map)
        #c_mat1=chunking_dot(c_mat1,bl_shrinkage_map)
        c_mat2=np.matmul(c_mat2,bl_shrinkage_map)

        c_mat1=np.transpose(c_mat1,(2,0,1))
        c_mat2=np.transpose(c_mat2,(2,0,1))

        c_mat1 = np.reshape(c_mat1, half_c_mat_shape)
        c_mat2 = np.reshape(c_mat2, half_c_mat_shape)

    c_mat = np.concatenate((c_mat1, c_mat2), axis=0)

    w_lut = c_mat * bl_pruning_mask
    #weights = [weights, w_lut]
    weights.extend([w_lut])
    #gammas = [gammas, bl_gamma]
    gammas.extend([bl_gamma])
    #pruning_masks = [pruning_masks, bl_pruning_mask]
    pruning_masks.extend([bl_pruning_mask])
    #rand_maps = [rand_maps, bl_rand_map]
    rand_maps.extend([bl_rand_map])
    #means = [means, bl_means]
    means.extend([bl_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])


    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    weights[0][0] = SignNumpy(weights[0][0])

    for i in range(1,5): #First layer not as LUTNet
        for j in range(32):
            weights[i][j] = SignNumpy(weights[i][j])

    # write header file
    with open('../codegen_output/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open('../codegen_output/weights.h', 'a') as f:
        f.write('//Generated weights for CIFAR-10\n')

    for layer_id in range(5):
        # generate weights
        if layer_id==0:  # first layer: fxp inputs and binary weights
            weights_per_act = 1
        else:
            weights_per_act = 32 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs

        dims = np.shape(weights[layer_id][0])
        if len(dims)==2:
            layer_type = "fc"
            word_length = dims[0]
            nfilters = dims[1]
        elif len(dims)==4:
            layer_type = "conv"
            word_length = dims[0]*dims[1]*dims[2]
            nfilters = dims[3]

#        for weight_id in range(weights_per_act):
#            mat = weights[layer_id][weight_id]
#            if layer_type=="fc":
#                mat_flat = mat.transpose(1,0).flatten()
#            elif layer_type=="conv":
#                mat_flat = mat.transpose(3,0,1,2).flatten()
#            else:
#                print("unknown weight format!")
#
#            with open('../src/weights.h', 'a') as f:
#                f.write('//Array shape: {}\n'.format(dims))
#                fold = (word_length-1)/32 + 1
#                f.write("const ap_uint<32> " + "weights_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters*fold) + "] = {")
#                bin_append = 0
#                for i, ele in enumerate(mat_flat):
#                    #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
#                    bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
#                    if (i % word_length == (word_length - 1)):
#                        mask = 0xFFFFFFFF
#                        for i_32b in range(fold):
#                            #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
#                            word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
#                            hex_word = '%X' % word
#                            if i_32b!=0:
#                                f.write(', ')
#                            f.write('0x' + hex_word)
#                        bin_append = 0
#                        if i != nfilters*word_length-1:
#                            f.write(', ')
#                f.write('};\n')

        if layer_id!=0:
            # generate verilog source file for LUTARRAY: Vivado HLS will take forever
            with open('../codegen_output/LUTARRAY_b0_MNIST_' + str(layer_id) + '.v', 'w') as v0:
                v0.write('`timescale 1 ns / 1 ps\n\n')
                v0.write('module LUTARRAY_b0 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
                for tm in range(nfilters):
                    v0.write(',\n        ap_return_' + str(tm))
                v0.write(');\n\n')
            with open('../codegen_output/LUTARRAY_b1_MNIST_' + str(layer_id) + '.v', 'w') as v1:
                v1.write('`timescale 1 ns / 1 ps\n\n')
                v1.write('module LUTARRAY_b1 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
                for tm in range(nfilters):
                    v1.write(',\n        ap_return_' + str(tm))
                v1.write(');\n\n')

            mat_flat = []       
            for weight_id in range(weights_per_act):
                mat = weights[layer_id][weight_id]
                pm = pruning_masks[layer_id]#.transpose(3,0,1,2).flatten()
                if layer_type=="fc":
                    mat = mat.transpose(1,0)
                    pm_flat = pm.transpose(1,0)
                elif layer_type=="conv":
                    mat = mat.transpose(3,0,1,2).reshape((nfilters, -1))
                    pm_flat = pm.transpose(3,0,1,2).reshape((nfilters, -1))
                else:
                    print("unknown weight format!")
                mat_flat.extend([mat])

            with open('../codegen_output/LUTARRAY_b0_MNIST_' + str(layer_id) + '.v', 'a') as v0:
                v0.write('\n\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_1_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_2_V;\n')
                v0.write('input  [' + str(word_length-1) + ':0] in_3_V;\n')
                for tm in range(nfilters):
                    v0.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v0.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v0.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v0.write('(' + str(int(mat_flat[16][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[17][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[18][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[19][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[20][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[21][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[22][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[23][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[24][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[25][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[26][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[27][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[28][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[29][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[30][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v0.write('(' + str(int(mat_flat[31][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']);\n')
                    v0.write('assign ap_return_' + str(tm) + ' = {')
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele == 0:
                            v0.write("1'b0")
                        elif ele == 1:
                            v0.write('tmp_' + str(tm) + '_' + str(ti))
                        else:
                            print("pruning mask elements must be binary!")
                        if ti != word_length-1:
                            v0.write(',')
                        else:
                            v0.write('};\n')
                v0.write('endmodule')
            with open('../codegen_output/LUTARRAY_b1_MNIST_' + str(layer_id) + '.v', 'a') as v1:
                v1.write('\n\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_1_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_2_V;\n')
                v1.write('input  [' + str(word_length-1) + ':0] in_3_V;\n')
                for tm in range(nfilters):
                    v1.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
                for tm in range(nfilters):
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele==1:
                            v1.write('wire tmp_' + str(tm) + '_' + str(ti) + ';\n')
                            v1.write('assign tmp_' + str(tm) + '_' + str(ti) + ' = ')
                            v1.write('(' + str(int(mat_flat[0][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[1][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[2][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[3][tm][ti])) + ' & in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[4][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[5][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[6][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[7][tm][ti])) + ' & in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[8][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[9][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[10][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[11][tm][ti])) + ' & ~in_V[' + str(ti) + '] & in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[12][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[13][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[14][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & in_3_V[' + str(ti) + ']) | ')
                            v1.write('(' + str(int(mat_flat[15][tm][ti])) + ' & ~in_V[' + str(ti) + '] & ~in_1_V[' + str(ti) + '] & ~in_2_V[' + str(ti) + '] & ~in_3_V[' + str(ti) + ']);\n')
                    v1.write('assign ap_return_' + str(tm) + ' = {')
                    for ti, ele in enumerate(pm_flat[tm]):
                        if ele == 0:
                            v1.write("1'b0")
                        elif ele == 1:
                            v1.write('tmp_' + str(tm) + '_' + str(ti))
                        else:
                            print("pruning mask elements must be binary!")
                        if ti != word_length-1:
                            v1.write(',')
                        else:
                            v1.write('};\n')
                v1.write('endmodule')

        # generate pruning mask (first layer only)
        if layer_id==0:
            #pruning_mask_flat = pruning_masks[layer_id].transpose(3,0,1,2).flatten()
            pruning_mask_flat = pruning_masks[layer_id].transpose(1,0).flatten()
            with open('../codegen_output/weights.h', 'a') as f:
                fold = int((word_length-1)/32 + 1)
                f.write("const ap_uint<32> " + "pruning_mask_" + layer_type + str(layer_id+1) + "_" + str(1) + "["+str(nfilters*fold) + "] = {")
                bin_append = 0
                for i, ele in enumerate(pruning_mask_flat):
                    #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                    bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
                    if (i % word_length == (word_length - 1)):
                        mask = 0xFFFFFFFF
                        for i_32b in range(fold):
                            #word = (bin_append>>(32*(fold-i_32b-1))) & mask # Big-endian: left-first word-push
                            word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                            hex_word = '%X' % word
                            if i_32b!=0:
                                f.write(', ')
                            f.write('0x' + hex_word)
                        bin_append = 0
                        if i != nfilters*word_length-1:
                            f.write(', ')
                f.write('};\n')
        # generate threshold
        if layer_id!=4: # the last layer does not need threshold
            use_popcount = not(layer_id==0)
            next_means_b0 = abs(means[layer_id][0])
            print(next_means_b0)
            next_means_b1 = abs(means[layer_id][1])
            print(next_means_b1)
            if layer_type=="conv":
                fanin = np.sum(pruning_masks[layer_id].reshape(-1,dims[3]),axis=0)
            elif layer_type=="fc":
                fanin = np.sum(pruning_masks[layer_id],axis=0)
            if layer_id!=0:
                fanin = fanin * abs(gammas[layer_id] * means[layer_id-1][0]) + fanin * abs(gammas[layer_id] * means[layer_id-1][1])
            thresholds = np.array(makeBNComplex(0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount))
            next_means_bn_b0 = np.array(makeBNComplex(next_means_b0, fanin, bn_betas[layer_id], bn_gammas[layer_id], bn_means[layer_id], bn_inv_stds[layer_id], usePopCount=use_popcount)) - thresholds

            with open('../codegen_output/weights.h', 'a') as f:
                f.write("const ap_fixed<24, 16> " + "thresh_" + layer_type + str(layer_id+1) + "["+str(len(thresholds))+"] = {")
                for i, ele in enumerate(thresholds):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
                f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id+1) + "["+str(len(next_means_bn_b0))+"] = {")
                for i, ele in enumerate(next_means_bn_b0):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
#        # generate next layer mean
#        if layer_id!=8:
#            with open('../src/weights.h', 'a') as f:
#                next_means_b0 = abs(means[layer_id][0])
#                next_means_b1 = abs(means[layer_id][1])
#                f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id+1) + "[2] = {")
#                f.write(str(next_means_b0))
#                f.write(','+ str(next_means_b1))
#                f.write('};\n')


        # generate random map
        for j in range(3):
            with open('../codegen_output/weights.h', 'a') as f:
                rand_map = rand_maps[layer_id][j].flatten().astype(np.uint32)
                f.write("const unsigned int " + "rand_map_" + str(j) + "_" + layer_type + str(layer_id+1) + "["+str(len(rand_map))+"] = {")
                for i, ele in enumerate(rand_map):
                    if i == 0:
                        f.write(str(ele))
                    else:
                        f.write(','+ str(ele))
                f.write('};\n')
        # generate alpha
        with open('../codegen_output/weights.h', 'a') as f:
            if layer_id!=0:
                alpha_b0 = abs(gammas[layer_id] * means[layer_id-1][0])
                alpha_b1 = abs(gammas[layer_id] * means[layer_id-1][1])
                f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(layer_id+1) + "[2] = {")
                f.write(str(alpha_b0))
                f.write(','+ str(alpha_b1))
                f.write('};\n')

            else:
                alpha_b0 = abs(gammas[layer_id])
                f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(layer_id+1) + "[1] = {")
                f.write(str(alpha_b0))
                f.write('};\n')




