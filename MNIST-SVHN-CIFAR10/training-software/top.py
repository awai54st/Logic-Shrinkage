from model_architectures import get_model
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

dataset = "CIFAR-10"

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


if mode == 'baseline':
	acc = lutnet_baseline(K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase) # Baseline LUTNet training without logic shrinkage
elif mode == 'ls':
	acc = lutnet_logic_shrinkage(K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase) # LUTNet with logic shrinkage
else:
	print("Mode should be either 'baseline' or 'ls'.")

