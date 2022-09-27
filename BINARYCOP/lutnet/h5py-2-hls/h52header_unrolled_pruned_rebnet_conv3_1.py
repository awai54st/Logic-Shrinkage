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
  thresholds = list(range(outs))
  for neuron in range(outs):
    # compute a preliminary threshold from the batchnorm parameters
    thres = mean[neuron] + ((after_bn_thres - beta[neuron]) / (abs(gamma[neuron]*invstd[neuron])))
    # turn threshold into "number of 1s" (popcount) instead of signed sum
    if usePopCount:
        thresholds[neuron] = (fanin[neuron] + thres) / 2
    else:
        thresholds[neuron] = thres

  return thresholds

if __name__ == "__main__":

    print("Loading the pretrained parameters...")
    
    targetDirBin = "pretrained_network"
    h5_file_name = targetDirBin+".h5"
    targetDirHLS = targetDirBin+"/hw"

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
    
    # conv layer 4
    prev_means = bl["model_weights"]["residual_sign_4"]["residual_sign_4"]["means:0"]
    
    # conv layer 5
    layer_id = 4
    
    bl_w1 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]).reshape(bl_w1.shape)
    bl_gamma = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"])
    
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]

    w_lut = [bl_w1*bl_pruning_mask, -bl_w1*bl_pruning_mask]
    weights.extend([w_lut])
    gammas.extend([bl_gamma])
    pruning_masks.extend([bl_pruning_mask])
    means.extend([bl_means])
    bn_betas.extend([bl_bn_beta])
    bn_gammas.extend([bl_bn_gamma])
    bn_means.extend([bl_bn_mean])
    bn_inv_stds.extend([bl_bn_inv_std])


    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    weights[0][0] = SignNumpy(weights[0][0])

    for j in range(2):
        weights[0][j] = SignNumpy(weights[0][j])

    # write header file
    with open(targetDirHLS+'/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open(targetDirHLS+'/weights.h', 'a') as f:
        f.write('//Generated weights for µ-CNV\n')

    weights_per_act = 2 # weights_per_act = #_of_bits_per_act x 2 ^ #_of_lut_inputs

    dims = np.shape(weights[0][0])
    if len(dims)==2:
        layer_type = "fc"
        word_length = dims[0]
        nfilters = dims[1]
    elif len(dims)==4:
        layer_type = "conv"
        word_length = dims[0]*dims[1]*dims[2]
        nfilters = dims[3]
        print("nfilters", nfilters)

    print("word_length", word_length)
    print("pruning masks", sum(pruning_masks[0].flatten())/len(pruning_masks[0].flatten()))


    for weight_id in range(weights_per_act):
        mat = weights[0][weight_id]
        pm = pruning_masks[0]#.transpose(3,0,1,2).flatten()
        if layer_type=="fc":
            mat_flat = mat.transpose(1,0).flatten()
        elif layer_type=="conv":
            mat_flat = mat.transpose(3,0,1,2).flatten()
        else:
            print("unknown weight format!")

    # generate verilog source file for LUTARRAY: Vivado HLS will take forever
    with open(targetDirHLS+'/XNORARRAY_b0_' + str(layer_id) + '.v', 'w') as v0:
        v0.write('`timescale 1 ns / 1 ps\n\n')
        v0.write('module XNORARRAY_b0 (\n        lut_out_63_V_write')
        for tm in range(nfilters):
            v0.write(',\n        ap_return_' + str(tm))
        v0.write(');\n\n')
    with open(targetDirHLS+'/XNORARRAY_b1_' + str(layer_id) + '.v', 'w') as v1:
        v1.write('`timescale 1 ns / 1 ps\n\n')
        v1.write('module XNORARRAY_b1 (\n        lut_out_63_V_write')
        for tm in range(nfilters):
            v1.write(',\n        ap_return_' + str(tm))
        v1.write(');\n\n')

    for weight_id in range(weights_per_act):
        #print(weights)
        mat = weights[0][weight_id]
        if layer_type=="fc":
            mat_flat = mat.transpose(1,0).flatten()
            pm_flat = pm.transpose(1,0)
        elif layer_type=="conv":
            mat_flat = mat.transpose(3,0,1,2).flatten()
            pm_flat = pm.transpose(3,0,1,2).reshape((nfilters, -1))
        else:
            print("unknown weight format!")

        with open(targetDirHLS+'/XNORARRAY_b0_' + str(layer_id) + '.v', 'a') as v:
            bin_append = 0
            for i, ele in enumerate(mat_flat):
                #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
                if (i % word_length == (word_length - 1)):
                    hex_word = '%X' % bin_append
                    v.write('parameter    ap_const_lv' + str(word_length) + '_' + str(weight_id) + '_' + str(round(i/word_length)-1) + ' = ' + str(word_length) + "'h" + hex_word + ';\n')
                    bin_append = 0
        with open(targetDirHLS+'/XNORARRAY_b1_' + str(layer_id) + '.v', 'a') as v:
            bin_append = 0
            for i, ele in enumerate(mat_flat):
                #bin_append = (bin_append << 1) | (int(ele) # left-first bit-push
                bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
                if (i % word_length == (word_length - 1)):
                    hex_word = '%X' % bin_append
                    v.write('parameter    ap_const_lv' + str(word_length) + '_' + str(weight_id) + '_' + str(round(i/word_length)-1) + ' = ' + str(word_length) + "'h" + hex_word + ';\n')
                    bin_append = 0

    with open(targetDirHLS+'/XNORARRAY_b0_' + str(layer_id) + '.v', 'a') as v0:
        v0.write('\n\n')
        v0.write('input  [' + str(word_length-1) + ':0] lut_out_63_V_write;\n')
        for tm in range(nfilters):
            v0.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
        for tm in range(nfilters):
            v0.write('assign ap_return_' + str(tm) + ' = (ap_const_lv' + str(word_length) + '_0_' + str(tm) + ' & lut_out_63_V_write) | (ap_const_lv' + str(word_length) + '_1_'+ str(tm) + ' & ~lut_out_63_V_write);\n')
        v0.write('endmodule')
    with open(targetDirHLS+'/XNORARRAY_b1_' + str(layer_id) + '.v', 'a') as v1:
        v1.write('\n\n')
        v1.write('input  [' + str(word_length-1) + ':0] lut_out_63_V_write;\n')
        for tm in range(nfilters):
            v1.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
        for tm in range(nfilters):
            v1.write('assign ap_return_' + str(tm) + ' = (ap_const_lv' + str(word_length) + '_0_' + str(tm) + ' & lut_out_63_V_write) | (ap_const_lv' + str(word_length) + '_1_'+ str(tm) + ' & ~lut_out_63_V_write);\n')
        v1.write('endmodule')

    # generate threshold
    use_popcount = False
    next_means_b0 = abs(means[0][0])
    print(next_means_b0)
    next_means_b1 = abs(means[0][1])
    print(next_means_b1)
    if layer_type=="conv":
        fanin = np.sum(pruning_masks[0].reshape(-1,dims[3]),axis=0)
        print(fanin)
    elif layer_type=="fc":
        fanin = np.sum(pruning_masks[0],axis=0)
    fanin = fanin * abs(gammas[0] * prev_means[0]) + fanin * abs(gammas[0] * prev_means[1])
    thresholds = np.array(makeBNComplex(0, fanin, bn_betas[0], bn_gammas[0], bn_means[0], bn_inv_stds[0], usePopCount=use_popcount))
    next_means_bn_b0 = np.array(makeBNComplex(next_means_b0, fanin, bn_betas[0], bn_gammas[0], bn_means[0], bn_inv_stds[0], usePopCount=use_popcount)) - thresholds
    
    with open(targetDirHLS+'/weights.h', 'a') as f:
        f.write("const ap_fixed<24, 16> " + "thresh_" + layer_type + str(layer_id) + "["+str(len(thresholds))+"] = {")
        for i, ele in enumerate(thresholds):
            if i == 0:
                f.write(str(ele))
            else:
                f.write(','+ str(ele))
        f.write('};\n')
        f.write("const ap_fixed<24, 16> " + "next_layer_means_" + layer_type + str(layer_id) + "["+str(len(next_means_bn_b0))+"] = {")
        for i, ele in enumerate(next_means_bn_b0):
            if i == 0:
                f.write(str(ele))
            else:
                f.write(','+ str(ele))
        f.write('};\n')
    
    # generate alpha
    with open(targetDirHLS+'/weights.h', 'a') as f:
        alpha_b0 = abs(gammas[0] * prev_means[0])
        alpha_b1 = abs(gammas[0] * prev_means[1])
        f.write("const ap_fixed<24, 16> " + "alpha_" + layer_type + str(layer_id) + "[2] = {")
        f.write(str(alpha_b0))
        f.write(','+ str(alpha_b1))
        f.write('};\n')

    # generate fan-in for PrunedPopCount
    with open(targetDirHLS+'/weights.h', 'a') as f:
        f.write("const ap_uint<16> " + "fanin" + "_" + layer_type + str(layer_id) + "["+str(nfilters)+"] = {")
        for tm in range(nfilters):
            if tm == 0:
                f.write(str(int(sum(pm_flat[tm]))))
            else:
                f.write(','+ str(int(sum(pm_flat[tm]))))
        f.write('};\n')
