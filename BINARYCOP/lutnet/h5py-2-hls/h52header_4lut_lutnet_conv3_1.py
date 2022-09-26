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
        #thresholds[neuron] = int((fanin[neuron] + thres) / 2)
        thresholds[neuron] = (fanin[neuron] + thres) / 2
    else:
        thresholds[neuron] = thres

  return thresholds

if __name__ == "__main__":

    print("Loading the pretrained parameters...")

    logic_shrinkage = False
    
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
    
    bl_c_param = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable_1:0"])

    bl_rand_map_0 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_0:0"])
    bl_rand_map_1 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_1:0"])
    bl_rand_map_2 = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["rand_map_2:0"])
    bl_pruning_mask = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["pruning_mask:0"]).reshape(bl_c_param[0].shape)
    print("pruning_mask ", sum(bl_pruning_mask.flatten())/len(bl_pruning_mask.flatten()))
    bl_gamma = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["Variable:0"])
 
    bl_shrinkage_map = np.array(bl["model_weights"]["binary_conv_5"]["binary_conv_5"]["shrinkage_map:0"])
    print("bl_shrinkage_map", sum(bl_shrinkage_map.flatten())/len(bl_shrinkage_map.flatten()))
   
    bl_bn_beta = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["beta:0"])
    bl_bn_gamma = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["gamma:0"])
    bl_bn_mean = np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_mean:0"])
    bl_bn_inv_std = 1/np.sqrt(np.array(bl["model_weights"]["batch_normalization_5"]["batch_normalization_5"]["moving_variance:0"])+batch_norm_eps)
    
    bl_means = bl["model_weights"]["residual_sign_5"]["residual_sign_5"]["means:0"]

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
    #print("c_mat", c_mat)

    w_lut = c_mat * bl_pruning_mask
    weights.extend([w_lut])
    gammas.extend([bl_gamma])
    pruning_masks.extend([bl_pruning_mask])
    bl_rand_map = [bl_rand_map_0, bl_rand_map_1, bl_rand_map_2]
    rand_maps.extend([bl_rand_map])
    means.extend([bl_means])
    bn_betas.extend([bl_bn_beta])
    bn_gammas.extend([bl_bn_gamma])
    bn_means.extend([bl_bn_mean])
    bn_inv_stds.extend([bl_bn_inv_std])


    print("Binarizing the pretrained parameters...")

    # Binarize the weights
    weights[0][0] = SignNumpy(weights[0][0])

    for j in range(32):
        weights[0][j] = SignNumpy(weights[0][j])

    # write header file
    with open(targetDirHLS+'/weights.h', 'w') as f:
        f.write('#pragma once\n')
    with open(targetDirHLS+'/weights.h', 'a') as f:
        f.write('//Generated weights for µ-CNV\n')

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
        print("nfilters", nfilters)

    for weight_id in range(weights_per_act):
        mat = weights[0][weight_id]
        if layer_type=="fc":
            mat_flat = mat.transpose(1,0).flatten()
        elif layer_type=="conv":
            mat_flat = mat.transpose(3,0,1,2).flatten()
        else:
            print("unknown weight format!")

        """
        with open(targetDirHLS+'/weights.h', 'a') as f:
            f.write('//Array shape: {}\n'.format(dims))
            fold = int((word_length-1)/32 + 1)
            f.write("const ap_uint<32> " + "weights_" + layer_type + str(layer_id+1) + "_" + str(weight_id+1) + "["+str(nfilters*fold) + "] = {")
            bin_append = 0
            for i, ele in enumerate(mat_flat):
                bin_append = bin_append | (int(ele) << (i % word_length)) # right-first bit-push
                if (i % word_length == (word_length - 1)):
                    mask = 0xFFFFFFFF
                    for i_32b in range(fold):
                        word = (bin_append>>(32*i_32b)) & mask # Little-endian: right-first word-push
                        hex_word = '%X' % word
                        if i_32b!=0:
                            f.write(', ')
                        f.write('0x' + hex_word)
                    bin_append = 0
                    if i != nfilters*word_length-1:
                        f.write(', ')
            f.write('};\n')
        """

    # generate verilog source file for LUTARRAY: Vivado HLS will take forever
    with open(targetDirHLS+'/LUTARRAY_b0_' + str(layer_id) + '.v', 'w') as v0:
        v0.write('`timescale 1 ns / 1 ps\n\n')
        v0.write('module LUTARRAY_b0 (\n        in_V,\n        in_1_V,\n        in_2_V,\n        in_3_V')
        for tm in range(nfilters):
            v0.write(',\n        ap_return_' + str(tm))
        v0.write(');\n\n')
    with open(targetDirHLS+'/LUTARRAY_b1_' + str(layer_id) + '.v', 'w') as v1:
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

    with open(targetDirHLS+'/LUTARRAY_b0_' + str(layer_id) + '.v', 'a') as v0:
        v0.write('\n\n')
        v0.write('input  [' + str(word_length-1) + ':0] in_V;\n')
        v0.write('input  [' + str(word_length-1) + ':0] in_1_V;\n')
        v0.write('input  [' + str(word_length-1) + ':0] in_2_V;\n')
        v0.write('input  [' + str(word_length-1) + ':0] in_3_V;\n')
        for tm in range(nfilters):
            v0.write('output  [' + str(word_length-1) + ':0] ap_return_' + str(tm) + ';\n')
        for tm in range(nfilters):
            #print("pm_flat[", tm, "] = \n", pm_flat[tm])
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
    with open(targetDirHLS+'/LUTARRAY_b1_' + str(layer_id) + '.v', 'a') as v1:
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
    use_popcount = False#not(layer_id==0)
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

    # generate random map
    for j in range(3):
        with open(targetDirHLS+'/weights.h', 'a') as f:
            #print(rand_maps[0][j])
            rand_map = rand_maps[0][j].flatten().astype(np.uint32)
            f.write("const unsigned int " + "rand_map_" + str(j) + "_" + layer_type + str(layer_id) + "["+str(len(rand_map))+"] = {")
            for i, ele in enumerate(rand_map):
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
