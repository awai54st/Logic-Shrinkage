from model_architectures import get_model
from rebnet_unpruned import rebnet_unpruned
from rebnet_pruned import rebnet_pruned
from lutnet_vanilla import lutnet_vanilla
from lutnet_logic_shrinkage import lutnet_logic_shrinkage
from logic_expansion_from_best_rebnet import logic_expansion_from_best_rebnet
from lutnet_bin_retraining_from_best_logic_shrinkage import lutnet_bin_retraining_from_best_logic_shrinkage
import sys
from logger import logger

mode = sys.argv[1].replace('\r', '')
K = int(sys.argv[2])
hidden_layer_pruning_percentage = float(sys.argv[3])
activation_pruning_percentage = float(sys.argv[4])
output_path = sys.argv[5].replace('\r', '')
custom_rand_seed = int(sys.argv[6])
training_phase = int(sys.argv[7])
dataset = sys.argv[8].replace('\r', '')

logger.info("mode is "+mode)
logger.info("K is "+str(K))
logger.info("hidden_layer_pruning_percentage is "+str(hidden_layer_pruning_percentage))
logger.info("activation_pruning_percentage is "+str(activation_pruning_percentage))
logger.info("output_path is "+output_path)
logger.info("custom_rand_seed is "+str(custom_rand_seed))
logger.info("training_phase is "+str(training_phase))
logger.info("dataset is "+dataset)

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
elif dataset == "BinaryCoP":
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
elif dataset == "mu-BinaryCoP":
	pruning_percentage = {
		b'binary_conv_1': -1,
		b'binary_conv_2': -1,
		b'binary_conv_3': -1,
		b'binary_conv_4': -1,
		b'binary_conv_5': hidden_layer_pruning_percentage,
		b'binary_dense_1': -1,
		b'binary_dense_2': -1,
	}
elif dataset == "MNIST":
	pruning_percentage = {
		b'binary_dense_1': -1,
		b'binary_dense_2': hidden_layer_pruning_percentage,
		b'binary_dense_3': hidden_layer_pruning_percentage,
		b'binary_dense_4': hidden_layer_pruning_percentage,
		b'binary_dense_5': -1,
	}
else:
	raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, BinaryCoP, mu-BinaryCoP].")

logger.info("pruning_percentage is "+str(pruning_percentage))

if mode == 'rebnet_unpruned': # Unpruned ReBNet
	acc = rebnet_unpruned(dataset, K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase)
elif mode == 'rebnet_pruned': # Pruned ReBNet
	acc = rebnet_pruned(dataset, K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase)
elif mode == 'lutnet_vanilla': # Vanilla LUTNet training without logic shrinkage
	acc = lutnet_vanilla(dataset, K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase)
elif mode == 'logic_expansion_from_best_rebnet': # Vanilla LUTNet logic expansion retraining using float32 only
	acc = logic_expansion_from_best_rebnet(dataset, K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase)
elif mode == 'lutnet_logic_shrinkage':
	acc = lutnet_logic_shrinkage(dataset, K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase) # LUTNet with logic shrinkage
elif mode == 'lutnet_bin_retraining_from_best_logic_shrinkage':
	acc = lutnet_bin_retraining_from_best_logic_shrinkage(dataset, K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase) # LUTNet with logic shrinkage
else:
	print("Mode should be either 'rebnet_unpruned', 'rebnet_pruned', 'lutnet_vanilla' or 'lutnet_logic_shrinkage'.")