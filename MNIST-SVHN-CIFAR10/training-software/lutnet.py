import os
from Binary import train_lutnet
from bnn_pruning import bnn_pruning 
from lutnet_init import lutnet_init
from shutil import copyfile
from logic_shrinkage import logic_shrinkage

def lutnet(K, get_model, pruning_percentage):

	bnnEpochs=100
	pruningEpochs=50
	lutnetFP32Epochs=50
	lutnetBINEpochs=50
	bnnBatchSize=100
	lutnetBatchSize=100 #avoid VRAM overflow

	# def train_lutnet(get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHINKAGE, trainable_means, Evaluate, epochs, batch_size)
	if not os.path.isfile('models/dummy.h5'):
		train_lutnet(get_model, pruning_percentage, K, True, True, False, True, False, False, True, False, 1, lutnetBatchSize)
		copyfile("models/2_residuals.h5", "models/dummy.h5")
	train_lutnet(get_model, pruning_percentage, K, True, True, False, False, False, False, True, False, bnnEpochs, bnnBatchSize)
	
	print ("Finished training bnn from scratch. ")

	bnn_pruning(get_model, pruning_percentage)                 
	train_lutnet(get_model, pruning_percentage, K, True, False, True, False, True, False, True, True, pruningEpochs, bnnBatchSize)
	
	print("Finished bnn pruning and retraining bnn. ")
	
	lutnet_init(K, get_model)
	train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, False, True, True, lutnetFP32Epochs, lutnetBatchSize)

	print("Finished logic expansion. ")

	logic_shrinkage(K)
	train_lutnet(get_model, pruning_percentage, K, True, False, False, True, False, True, True, True, lutnetFP32Epochs, lutnetBatchSize)
	acc = train_lutnet(get_model, pruning_percentage, K, True, False, False, True, True, True, False, True, lutnetBINEpochs, lutnetBatchSize)
	
	print(acc)

	return acc

