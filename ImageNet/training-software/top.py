from model_architectures import get_model
#from model_architectures import pruning_percentage
#from model_architectures import K
#from model_architectures import num_act_pruned
from lutnet import lutnet
from lutnet_baseline import lutnet_baseline
from lutnet_logic_shrinkage import lutnet_logic_shrinkage
import sys

mode = sys.argv[1]
K = int(sys.argv[2])
hidden_layer_pruning_percentage = float(sys.argv[3])
activation_pruning_percentage = float(sys.argv[4])
output_path = sys.argv[5]
custom_rand_seed = int(sys.argv[6])
training_phase = int(sys.argv[7])

dataset = "IMAGENET"

if dataset in ["CIFAR-10", "SVHN"]:
	pruning_percentage = {
		b'binary_conv_1': -1,
		b'binary_conv_2': -1,
		b'binary_conv_3': -1,
		b'binary_conv_4': -1,
		b'binary_conv_5': -1,
		b'binary_conv_6': hidden_layer_pruning_percentage,
		b'binary_dense_1': -1,
		b'binary_dense_2': -1,
		b'binary_dense_3': -1,
	}
elif dataset == "MNIST":
	pruning_percentage = {
		b'binary_dense_1': -1,
		b'binary_dense_2': hidden_layer_pruning_percentage,
		b'binary_dense_3': hidden_layer_pruning_percentage,
		b'binary_dense_4': hidden_layer_pruning_percentage,
		b'binary_dense_5': -1,
	}
elif dataset == "IMAGENET":
	pruning_percentage = {
		'binary_conv_1': -1,
		'binary_conv_2': -1,
		'binary_conv_3': -1, 
		'binary_conv_4': -1, 
		'binary_conv_5': -1,
		'binary_conv_6': -1,
		'binary_conv_7': -1,
		'binary_conv_8': -1,
		'binary_conv_9': -1,
		'binary_conv_10': -1,
		'binary_conv_11': -1,
		'binary_conv_12': -1,
		'binary_conv_13': -1,
		'binary_conv_14': -1,
		'binary_conv_15': -1,
		'binary_conv_16': -1,
		'binary_conv_17': -1,
		'binary_dense_1': -1,
	}


#acc = lutnet(K, get_model, pruning_percentage)
if mode == 'baseline':
	acc = lutnet_baseline(K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase) # Baseline LUTNet training without logic shrinkage
elif mode == 'ls':
	acc = lutnet_logic_shrinkage(K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase) # LUTNet with logic shrinkage
else:
	print("Mode should be either 'baseline' or 'ls'.")

