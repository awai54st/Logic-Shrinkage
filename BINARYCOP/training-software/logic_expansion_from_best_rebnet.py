import os
from binary import train_lutnet
from bnn_pruning import bnn_pruning 
from lutnet_init import lutnet_init
from shutil import copyfile
from logic_shrinkage import logic_shrinkage
from logger import logger

def logic_expansion_from_best_rebnet(dataset, K, get_model, pruning_percentage, output_path, custom_rand_seed, training_phase):
	
	bnnEpochs=75
	pruningEpochs=50
	lutnetFP32Epochs=50
	lutnetBINEpochs=50
	bnnBatchSize=100
	lutnetBatchSize=100 #avoid VRAM overflow
	
	# bnnEpochs=1
	# pruningEpochs=1
	# lutnetFP32Epochs=1
	# lutnetBINEpochs=1
	# bnnBatchSize=100
	# lutnetBatchSize=100 #avoid VRAM overflow

	if training_phase != 0 and not os.path.isfile(output_path + '/dummy.h5'):
		logger.info("Started dummy generation. ")
		Train = True; REG = True; Retrain = False; LUT = True; BINARY = False; LOGIC_SHRINKAGE = False; trainable_means = True; Evaluate = False
		train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, 1, lutnetBatchSize, output_path, custom_rand_seed)
		copyfile(output_path + "/2_residuals.h5", output_path + "/dummy.h5")
		logger.info("Finished dummy generation. ")

	if training_phase == 2:
	
		logger.info("Started lutnet initialisation (training phase 2/2 part 1/3)\n ")		
		lutnet_init(dataset, K, get_model, output_path, custom_rand_seed)
		logger.info("Finished lutnet initialisation (training phase 2/2 part 1/3)\n ")		

		logger.info("Started lutnet float32 training (training phase 2/2 part 2/3)\n ")		
		Train = True; REG = False; Retrain = False; LUT = True; BINARY = False; LOGIC_SHRINKAGE = False; trainable_means = True; Evaluate = True
		train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, lutnetFP32Epochs, lutnetBatchSize, output_path, custom_rand_seed)
		logger.info("Finished lutnet float32 training (training phase 2/2 part 2/3)\n ")		

	elif training_phase == 0: # evaluation mode

		logger.info("Started evaluation. ")
		Train = False; REG = False; Retrain = False; LUT = True; BINARY = True; LOGIC_SHRINKAGE = False; trainable_means = False; Evaluate = True
		acc = train_lutnet(dataset, get_model, pruning_percentage, K, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, 0, 0, output_path, custom_rand_seed)
		logger.info("Finished evaluation. ")

	logger.info("Accuracy is "+str(acc))

	return acc

