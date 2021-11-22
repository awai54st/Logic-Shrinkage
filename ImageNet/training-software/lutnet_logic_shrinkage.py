import os
from Binary import train_lutnet
from bnn_pruning import bnn_pruning 
from lutnet_init import lutnet_init
from shutil import copyfile
from logic_shrinkage import logic_shrinkage

def lutnet_logic_shrinkage(K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase):

	bnnEpochs=256#200
	pruningEpochs=64#32
	lutnetFP32Epochs=32#50
	lutnetBINEpochs=64#200
	lutnetLSEpochs=8#20
	bnnBatchSize=256#100
	lutnetBatchSize=32#64#100 #avoid VRAM overflow

	# def train_lutnet(get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHINKAGE, trainable_means, Evaluate, epochs, batch_size)
	if training_phase != 0 and not os.path.isfile(output_path + '/dummy.h5'):
		train_lutnet(get_model, pruning_percentage, K, True, True, False, True, False, False, True, False, 1, lutnetBatchSize, output_path, custom_rand_seed)
		copyfile(output_path + "/2_residuals.h5", output_path + "/dummy.h5")

	if training_phase == 1:

		train_lutnet(get_model, pruning_percentage, K, True, True, False, False, False, False, True, False, bnnEpochs, bnnBatchSize, output_path, custom_rand_seed)
		
		print ("Finished training bnn from scratch. ")
	
		bnn_pruning(get_model, pruning_percentage, output_path)                 
		acc = train_lutnet(get_model, pruning_percentage, K, True, False, True, False, True, False, True, True, pruningEpochs, bnnBatchSize, output_path, custom_rand_seed)
		
		print("Finished bnn pruning and retraining bnn. ")
	
	elif training_phase == 2:
	
		lutnet_init(K, get_model, output_path, custom_rand_seed)
		train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, False, True, True, lutnetFP32Epochs, lutnetBatchSize, output_path, custom_rand_seed)
	
		print("Finished logic expansion. ")
	
		## Without retraining
		#logic_shrinkage(get_model, K, activation_pruning_percentage, output_path, custom_rand_seed)

		# With retraining
		logic_shrinkage(get_model, K, activation_pruning_percentage*0.33, output_path, custom_rand_seed)
		train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, True, True, True, lutnetLSEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		logic_shrinkage(get_model, K, activation_pruning_percentage*0.67, output_path, custom_rand_seed)
		train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, True, True, True, lutnetLSEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		logic_shrinkage(get_model, K, activation_pruning_percentage, output_path, custom_rand_seed)

		########
		train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, True, True, True, lutnetFP32Epochs, lutnetBatchSize, output_path, custom_rand_seed)
		acc = train_lutnet(get_model, pruning_percentage, K, True, False, False, True, True, True, False, True, lutnetBINEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		
		print(acc)

	elif training_phase == 0: # evaluation mode

		acc = train_lutnet(get_model, pruning_percentage, K, False, False, False, True, True, True, False, True, 0, 0, output_path, custom_rand_seed)

	return acc

