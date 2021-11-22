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
    prev_means = []
    pruning_masks = []
    rand_maps = []
    bn_betas = []
    bn_gammas = []
    bn_means = []
    bn_inv_stds = []
    
    # conv layer 12

    bl_c_param = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)

    # Bi-Real Net scaling factors
    bl_gamma = np.mean(abs(bl_c_param),axis=(1,2,3, 4),keepdims=False)
    #bl_gamma = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["Variable:0"])
 
    bl_shrinkage_map = np.array(bl["model_weights"]["binary_conv_12"]["binary_conv_12"]["shrinkage_map:0"])
   
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_12"]["batch_normalization_12"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_12"]["batch_normalization_12"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_12"]["batch_normalization_12"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_12"]["batch_normalization_12"]["moving_variance:0"])+batch_norm_eps)
    
    bl_prev_means = bl["model_weights"]["residual_sign_11"]["residual_sign_11"]["means:0"]
    bl_means = bl["model_weights"]["residual_sign_12"]["residual_sign_12"]["means:0"]

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

    
    #w_lut = [bl_w1, bl_w2, bl_w3, bl_w4, bl_w5, bl_w6, bl_w7, bl_w8]
#    w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask,bl_w9*bl_pruning_mask, bl_w10*bl_pruning_mask, bl_w11*bl_pruning_mask, bl_w12*bl_pruning_mask, bl_w13*bl_pruning_mask, bl_w14*bl_pruning_mask, bl_w15*bl_pruning_mask, bl_w16*bl_pruning_mask,bl_w17*bl_pruning_mask, bl_w18*bl_pruning_mask, bl_w19*bl_pruning_mask, bl_w20*bl_pruning_mask, bl_w21*bl_pruning_mask, bl_w22*bl_pruning_mask, bl_w23*bl_pruning_mask, bl_w24*bl_pruning_mask,bl_w25*bl_pruning_mask, bl_w26*bl_pruning_mask, bl_w27*bl_pruning_mask, bl_w28*bl_pruning_mask, bl_w29*bl_pruning_mask, bl_w30*bl_pruning_mask, bl_w31*bl_pruning_mask, bl_w32*bl_pruning_mask]
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
    prev_means.extend([bl_prev_means])
    #bn_betas = [bn_betas, bl_bn_beta]
    bn_betas.extend([bl_bn_beta])
    #bn_gammas = [bn_gammas, bl_bn_gamma]
    bn_gammas.extend([bl_bn_gamma])
    #bn_means = [bn_means, bl_bn_mean]
    bn_means.extend([bl_bn_mean])
    #bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    bn_inv_stds.extend([bl_bn_inv_std])

   
    ## dense layer 1
    #
    #bl_w1 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_1:0"])
    ##bl_w2 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_2:0"])
    ##bl_w3 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_3:0"])
    ##bl_w4 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_4:0"])
    ##bl_w5 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_5:0"])
    ##bl_w6 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_6:0"])
    ##bl_w7 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_7:0"])
    ##bl_w8 = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable_8:0"])
    #bl_rand_map = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["rand_map_0:0"])
    #bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["pruning_mask:0"]).reshape(bl_w1.shape)
    #bl_gamma = np.array(bl["model_weights"]["binary_dense_1"]["binary_dense_1"]["Variable:0"])
    #
    #bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["beta:0"])
    #bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["gamma:0"])
    #bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_mean:0"])
    #bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_7"]["batch_normalization_7"]["moving_variance:0"])+batch_norm_eps)
    #
    #bl_means = bl["model_weights"]["residual_sign_7"]["residual_sign_7"]["means:0"]
    #
    ##w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    #w_lut = [bl_w1]
    ##weights = [weights, w_lut]
    #weights.extend([w_lut])
    ##gammas = [gammas, bl_gamma]
    #gammas.extend([bl_gamma])
    ##pruning_masks = [pruning_masks, bl_pruning_mask]
    #pruning_masks.extend([bl_pruning_mask])
    ##rand_maps = [rand_maps, bl_rand_map]
    #rand_maps.extend([bl_rand_map])
    ##means = [means, bl_means]
    #means.extend([bl_means])
    ##bn_betas = [bn_betas, bl_bn_beta]
    #bn_betas.extend([bl_bn_beta])
    ##bn_gammas = [bn_gammas, bl_bn_gamma]
    #bn_gammas.extend([bl_bn_gamma])
    ##bn_means = [bn_means, bl_bn_mean]
    #bn_means.extend([bl_bn_mean])
    ##bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    #bn_inv_stds.extend([bl_bn_inv_std])


  
    ## dense layer 2
    #
    #bl_w1 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_1:0"])
    ##bl_w2 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_2:0"])
    ##bl_w3 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_3:0"])
    ##bl_w4 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_4:0"])
    ##bl_w5 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_5:0"])
    ##bl_w6 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_6:0"])
    ##bl_w7 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_7:0"])
    ##bl_w8 = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable_8:0"])
    #bl_rand_map = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["rand_map_0:0"])
    #bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["pruning_mask:0"]).reshape(bl_w1.shape)
    #bl_gamma = np.array(bl["model_weights"]["binary_dense_2"]["binary_dense_2"]["Variable:0"])
    #
    #bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["beta:0"])
    #bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["gamma:0"])
    #bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_mean:0"])
    #bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_8"]["batch_normalization_8"]["moving_variance:0"])+batch_norm_eps)
    #
    #bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
    #
    ##w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    #w_lut = [bl_w1]
    ##weights = [weights, w_lut]
    #weights.extend([w_lut])
    ##gammas = [gammas, bl_gamma]
    #gammas.extend([bl_gamma])
    ##pruning_masks = [pruning_masks, bl_pruning_mask]
    #pruning_masks.extend([bl_pruning_mask])
    ##rand_maps = [rand_maps, bl_rand_map]
    #rand_maps.extend([bl_rand_map])
    ##means = [means, bl_means]
    #means.extend([bl_means])
    ##bn_betas = [bn_betas, bl_bn_beta]
    #bn_betas.extend([bl_bn_beta])
    ##bn_gammas = [bn_gammas, bl_bn_gamma]
    #bn_gammas.extend([bl_bn_gamma])
    ##bn_means = [bn_means, bl_bn_mean]
    #bn_means.extend([bl_bn_mean])
    ##bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    #bn_inv_stds.extend([bl_bn_inv_std])


    ## dense layer 3
    #
    #bl_w1 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_1:0"])
    ##bl_w2 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_2:0"])
    ##bl_w3 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_3:0"])
    ##bl_w4 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_4:0"])
    ##bl_w5 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_5:0"])
    ##bl_w6 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_6:0"])
    ##bl_w7 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_7:0"])
    ##bl_w8 = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable_8:0"])
    #bl_rand_map = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["rand_map_0:0"])
    #bl_pruning_mask = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["pruning_mask:0"]).reshape(bl_w1.shape)
    #bl_gamma = np.array(bl["model_weights"]["binary_dense_3"]["binary_dense_3"]["Variable:0"])
    #
    #bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["beta:0"])
    #bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["gamma:0"])
    #bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_mean:0"])
    #bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_9"]["batch_normalization_9"]["moving_variance:0"])+batch_norm_eps)
    #
    ##bl_means = bl["model_weights"]["residual_sign_8"]["residual_sign_8"]["means:0"]
    #
    ##w_lut = [bl_w1*bl_pruning_mask, bl_w2*bl_pruning_mask, bl_w3*bl_pruning_mask, bl_w4*bl_pruning_mask, bl_w5*bl_pruning_mask, bl_w6*bl_pruning_mask, bl_w7*bl_pruning_mask, bl_w8*bl_pruning_mask]
    #w_lut = [bl_w1]
    ##weights = [weights, w_lut]
    #weights.extend([w_lut])
    ##gammas = [gammas, bl_gamma]
    #gammas.extend([bl_gamma])
    ##pruning_masks = [pruning_masks, bl_pruning_mask]
    #pruning_masks.extend([bl_pruning_mask])
    ##rand_maps = [rand_maps, bl_rand_map]
    #rand_maps.extend([bl_rand_map])
    ##means = [means, bl_means]
    ##means.extend(bl_means)
    ##bn_betas = [bn_betas, bl_bn_beta]
    #bn_betas.extend([bl_bn_beta])
    ##bn_gammas = [bn_gammas, bl_bn_gamma]
    #bn_gammas.extend([bl_bn_gamma])
    ##bn_means = [bn_means, bl_bn_mean]
    #bn_means.extend([bl_bn_mean])
    ##bn_inv_stds = [bn_inv_stds, bl_bn_inv_std]
    #bn_inv_stds.extend([bl_bn_inv_std])


    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    for i in range(32):
        weights[0][i] = SignNumpy(weights[0][i])

    # write header file
    with open('../codegen_output/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open('../codegen_output/weights.h', 'a') as f:
        f.write('//Generated weights for IMAGENET\n')

    weights_per_act = 32 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs

    dims = np.shape(weights[0][0])
    if len(dims)==2:
        layer_type = "fc"
        word_length = dims[0]
        nfilters = dims[1]
    elif len(dims)==4:
        layer_type = "conv"
        word_length = dims[0]*dims[1]*dims[2]
        nfilters = dims[3]

    # generate verilog source file for LUTARRAY: Vivado HLS will take forever
    with open('../codegen_output/LUTARRAY_b0_' + str(0) + '.v', 'w') as v0:
        v0.write('`timescale 1 ns / 1 ps\n\n')
        v0.write('module LUTARRAY_b0 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
        for tm in range(nfilters):
            v0.write(',\n        ap_return_' + str(tm))
        v0.write(');\n\n')
    with open('../codegen_output/LUTARRAY_b1_' + str(0) + '.v', 'w') as v1:
        v1.write('`timescale 1 ns / 1 ps\n\n')
        v1.write('module LUTARRAY_b1 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
        for tm in range(nfilters):
            v1.write(',\n        ap_return_' + str(tm))
        v1.write(');\n\n')

    mat_flat = []       
    for weight_id in range(weights_per_act):
        mat = weights[0][weight_id]
        pm = pruning_masks[0]#.transpose(3,0,1,2).flatten()
        if layer_type=="fc":
            mat = mat.transpose(1,0)
            pm_flat = pm.transpose(1,0)
        elif layer_type=="conv":
            mat = mat.transpose(3,0,1,2).reshape((nfilters, -1))
            pm_flat = pm.transpose(3,0,1,2).reshape((nfilters, -1))
        else:
            print("unknown weight format!")
        mat_flat.extend([mat])

    with open('../codegen_output/LUTARRAY_b0_' + str(0) + '.v', 'a') as v0:
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
    with open('../codegen_output/LUTARRAY_b1_' + str(0) + '.v', 'a') as v1:
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

    # generate threshold
    use_popcount = True
    next_means_b0 = abs(means[0][0])
    print(next_means_b0)
    next_means_b1 = abs(means[0][1])
    print(next_means_b1)
    if layer_type=="conv":
        fanin = np.sum(pruning_masks[0].reshape(-1,dims[3]),axis=0)
    elif layer_type=="fc":
        fanin = np.sum(pruning_masks[0],axis=0)
    #if layer_id!=0:
    #    fanin = fanin * abs(gammas[0] * prev_means[0][0]) + fanin * abs(gammas[0] * prev_means[0][1])
    thresholds = np.array(makeBNComplex(0, fanin, bn_betas[0], bn_gammas[0], bn_means[0], bn_inv_stds[0], usePopCount=use_popcount))
    next_means_bn_b0 = np.array(makeBNComplex(next_means_b0, fanin, bn_betas[0], bn_gammas[0], bn_means[0], bn_inv_stds[0], usePopCount=use_popcount)) - thresholds

    with open('../codegen_output/weights.h', 'a') as f:
        f.write("const ap_fixed<24, 16> " + "thresh_" + layer_type + str(1) + "["+str(len(thresholds))+"] = {")
        for i, ele in enumerate(thresholds):
            if i == 0:
                f.write(str(ele))
            else:
                f.write(','+ str(ele))
        f.write('};\n')
        f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(1) + "["+str(len(next_means_bn_b0))+"] = {")
        for i, ele in enumerate(next_means_bn_b0):
            if i == 0:
                f.write(str(ele))
            else:
                f.write(','+ str(ele))
        f.write('};\n')

    # generate random map
    for j in range(3):
        with open('../codegen_output/weights.h', 'a') as f:
            rand_map = rand_maps[0][j].flatten().astype(np.uint32)
            f.write("const unsigned int " + "rand_map_" + str(j) + "_" + layer_type + str(1) + "["+str(len(rand_map))+"] = {")
            for i, ele in enumerate(rand_map):
                if i == 0:
                    f.write(str(ele))
                else:
                    f.write(','+ str(ele))
            f.write('};\n')
    # generate alpha
    with open('../codegen_output/weights.h', 'a') as f:
        alpha_b0 = abs(gammas[0][0] * prev_means[0][0])
        alpha_b1 = abs(gammas[0][1] * prev_means[0][1])
        f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(1) + "[2] = {")
        f.write(str(alpha_b0))
        f.write(','+ str(alpha_b1))
        f.write('};\n')




