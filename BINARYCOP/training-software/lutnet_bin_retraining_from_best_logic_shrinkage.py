import os
from binary import train_lutnet
from bnn_pruning import bnn_pruning 
from lutnet_init import lutnet_init
from shutil import copyfile
from logic_shrinkage import logic_shrinkage
from logger import logger

def lutnet_bin_retraining_from_best_logic_shrinkage(dataset, K, get_model, pruning_percentage, activation_pruning_percentage, output_path, custom_rand_seed, training_phase):

	bnnEpochs=75
	pruningEpochs=50
	lutnetFP32Epochs=50
	lutnetBINEpochs=200
	lutnetLSEpochs=20
	bnnBatchSize=100
	lutnetBatchSize=100 #avoid VRAM overflow

	# bnnEpochs=1
	# pruningEpochs=1
	# lutnetFP32Epochs=1
	# lutnetBINEpochs=1
	# lutnetLSEpochs=1
	# bnnBatchSize=100
	# lutnetBatchSize=100 #avoid VRAM overflow
	

	# def train_lutnet(get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHINKAGE, trainable_means, Evaluate, epochs, batch_size)
	# if training_phase != 0 and not os.path.isfile(output_path + '/dummy.h5'):
	# 	logger.info("Started dummy generation. ")
	# 	Train = True; REG = True; Retrain = False; LUT = True; BINARY = False; LOGIC_SHRINKAGE = False; trainable_means = True; Evaluate = False
	# 	train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, 1, lutnetBatchSize, output_path, custom_rand_seed)
	# 	copyfile(output_path + "/2_residuals.h5", output_path + "/dummy.h5")
	# 	logger.info("Finished dummy generation. ")
	
	if training_phase == 2:

		# logger.info("Started logic shrinkage (training phase 2/2 part 3/3)\n ")		
		# retraining = True
		# logger.info("Retraining is "+str(retraining))
		# Train = True; REG = False; Retrain = False; LUT = True; BINARY = False; LOGIC_SHRINKAGE = True; trainable_means = True; Evaluate = True

		# if retraining:
		# 	# With retraining
		# 	logic_shrinkage(dataset, get_model, K, activation_pruning_percentage*0.33, output_path, custom_rand_seed)
		# 	train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, lutnetLSEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		# 	logic_shrinkage(dataset, get_model, K, activation_pruning_percentage*0.67, output_path, custom_rand_seed)
		# 	train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, lutnetLSEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		# 	logic_shrinkage(dataset, get_model, K, activation_pruning_percentage, output_path, custom_rand_seed)
		# else:
		# 	# Without retraining
		# 	logic_shrinkage(dataset, get_model, K, activation_pruning_percentage, output_path, custom_rand_seed)

		# acc = train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, lutnetFP32Epochs, lutnetBatchSize, output_path, custom_rand_seed)
		# logger.info("Finished logic shrinkage (training phase 2/2 part 3/3)\n ")
		
		Train = True; REG = False; Retrain = False; LUT = True; BINARY = True; LOGIC_SHRINKAGE = True; trainable_means = False; Evaluate = True
		acc = train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, lutnetBINEpochs, lutnetBatchSize, output_path, custom_rand_seed)
		
		logger.info("Accuracy is "+str(acc))

	elif training_phase == 0: # evaluation mode
		logger.info("Started lutnet evaluation (training phase 0/2 part 1/1)\n ")
		Train = False; REG = False; Retrain = False; LUT = True; BINARY = True; LOGIC_SHRINKAGE = True; trainable_means = False; Evaluate = True
		acc = train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, 0, 0, output_path, custom_rand_seed)
		logger.info("Finished lutnet evaluation (training phase 0/2 part 1/1)\n ")

	else:
		raise("training_phase should be one of the following: [0, 1, 2].")

	return acc