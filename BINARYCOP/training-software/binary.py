import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
from tensorflow.keras.datasets import cifar10,mnist
from generate_binarycop_dataset import get_binarycop_dataset, get_image, write_image_bin_file
import tensorflow.keras.utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from binarization_utils import *
from logger import logger

from shutil import copyfile
import h5py


def load_svhn(path_to_dataset):
	import scipy.io as sio
	train=sio.loadmat(path_to_dataset+'/train.mat')
	test=sio.loadmat(path_to_dataset+'/test.mat')
	extra=sio.loadmat(path_to_dataset+'/extra.mat')
	X_train=np.transpose(train['X'],[3,0,1,2])
	y_train=train['y']-1
	
	X_test=np.transpose(test['X'],[3,0,1,2])
	y_test=test['y']-1
	
	X_extra=np.transpose(extra['X'],[3,0,1,2])
	y_extra=extra['y']-1
	
	X_train=np.concatenate((X_train,X_extra),axis=0)
	y_train=np.concatenate((y_train,y_extra),axis=0)
	
	return (X_train,y_train),(X_test,y_test)

def train_lutnet(dataset, get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, epochs, batch_size, output_path='models', custom_rand_seed=0):

	logger.info('Train is '+str(Train))
	logger.info('REG is '+str(REG))
	logger.info('Retrain is '+str(Retrain))
	logger.info('LUT is '+str(LUT))
	logger.info('BINARY is '+str(BINARY))
	logger.info('LOGIC_SHRINKAGE is '+str(LOGIC_SHRINKAGE))
	logger.info('trainable_means is '+str(trainable_means))
	logger.info('Evaluate is '+str(Evaluate))

	print('Train is '+str(Train))
	print('REG is '+str(REG))
	print('Retrain is '+str(Retrain))
	print('LUT is '+str(LUT))
	print('BINARY is '+str(BINARY))
	print('LOGIC_SHRINKAGE is '+str(LOGIC_SHRINKAGE))
	print('trainable_means is '+str(trainable_means))
	print('Evaluate is '+str(Evaluate))

	if dataset=="MNIST":
		(X_train, y_train), (X_test, y_test) = mnist.load_data()
		# convert class vectors to binary class matrices
		X_train = X_train.reshape(-1,784)
		X_test = X_test.reshape(-1,784)
		use_generator=False
	elif dataset=="CIFAR-10":
		use_generator=True
		(X_train, y_train), (X_test, y_test) = cifar10.load_data()
	elif dataset=="SVHN":
		use_generator=True
		(X_train, y_train), (X_test, y_test) = load_svhn('./svhn_data')
	elif dataset=="BinaryCoP" or dataset=="mu-BinaryCoP":
		use_generator=False	
		(X_train, y_train), (X_test, y_test) = get_binarycop_dataset()
	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, BinaryCoP, mu-BinaryCoP].")
	
	if dataset in ["MNIST", "CIFAR-10", "SVHN"]:
		X_train=X_train.astype(np.float32)
		X_test=X_test.astype(np.float32)
		Y_train = to_categorical(y_train, 10)
		Y_test = to_categorical(y_test, 10)
		X_train /= 256
		X_test /= 256
		X_train=2*X_train-1
		X_test=2*X_test-1
	elif dataset in ["BinaryCoP", "mu-BinaryCoP"]:
		X_train=X_train.astype(np.float32)
		X_test=X_test.astype(np.float32)
		# BinaryCoP has 4 classes
		Y_train = to_categorical(y_train, 4)
		Y_test = to_categorical(y_test, 4)
		X_train /= 256
		X_test /= 256
		X_train=2*X_train-1
		X_test=2*X_test-1
	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, BinaryCoP, mu-BinaryCoP].")
	
	if not Evaluate: 
		logger.info("X_train shape:"+str(X_train.shape))
		logger.info(str(X_train.shape[0])+" train samples")
	logger.info(str(X_test.shape[0])+" test samples")

	if Train:
		if not(os.path.exists(output_path)):
			os.mkdir(output_path)
		for resid_levels in range(2,3):
			logger.info("training with %d levels"%resid_levels)
			print("training with %d levels"%resid_levels)
			sess=K.get_session()
			model=get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed)
			#model.summary()
	
			#gather all binary dense and binary convolution layers:
			binary_layers=[]
			for l in model.layers:
				if isinstance(l,binary_dense) or isinstance(l,binary_conv):
					binary_layers.append(l)
	
			#gather all residual binary activation layers:
			resid_bin_layers=[]
			for l in model.layers:
				if isinstance(l,Residual_sign):
					resid_bin_layers.append(l)
			lr=0.01
			decay=1e-6
	
			weights_path = output_path + '/'+str(resid_levels)+'_residuals.h5'
			if Retrain: #pruned retraining
				#weights_path='models/pretrained_pruned.h5'
				model.load_weights(weights_path)
			elif LUT and not REG: #logic expansion
				#weights_path='models/pretrained_bin.h5'
				model.load_weights(weights_path)

			opt = keras.optimizers.Adam(lr=lr,decay=decay)#SGD(lr=lr,momentum=0.9,decay=1e-5)
			model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	
	
			cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_acc', save_best_only=True)
			if use_generator:
				if dataset in ["CIFAR-10"]:
					horizontal_flip=True
				if dataset in ["SVHN","BinaryCoP", "mu-BinaryCoP"]:
					horizontal_flip=False
				datagen = ImageDataGenerator(
					width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
					height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
					horizontal_flip=horizontal_flip)  # randomly flip images
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]/batch_size,
				epochs=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
	
			else:
				history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
			dic={'hard':history.history}
			foo=open(output_path + '/history_'+str(resid_levels)+'_residuals.pkl','wb')
			pickle.dump(dic,foo)
			foo.close()
		
		tf.Session().close()
		K.clear_session()

	if Evaluate:


		for resid_levels in range(2,3):
			
			weights_path = output_path + '/'+str(resid_levels)+'_residuals.h5'

			# # Copy 2_residuals.h5 file to modify stored parameters to run inference on
			# copyfile(weights_path, output_path + "/debug.h5") # create pretrained.h5 using datastructure from dummy.h5
			# source = h5py.File(weights_path, 'r')
			# dest = h5py.File(output_path + "/debug.h5", 'r+')

			# # Fill gamma with 1
			# src_gamma = source["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"]
			# dest_gamma = dest["model_weights"]["binary_conv_1"]["binary_conv_1"]["Variable:0"]
			# temp = np.array(dest_gamma)
			# #temp.fill(1.0)
			# dest_gamma[...] = temp
			# print("dest_gamma ", dest_gamma)

			# # Use the modified h5 file to run inference on
			# weights_path = output_path + "/debug.h5"

			# Evaluate the model
			model=get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed)
			model.load_weights(weights_path)
			opt = keras.optimizers.Adam()
			model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
			model.summary()
			score=model.evaluate(X_test,Y_test,verbose=0)

			# # Make an auxiliary model that exposes the output from the intermediate layer of interest
			# aux_model_bin_conv_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[0].output)
			# aux_model_bnorm_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[1].output)
			
			# aux_model_res_1_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output[0])
			# aux_model_res_1_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[2].output[1])
			
			# aux_model_bin_conv_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[3].output)
			# aux_model_res_2_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[6].output[0])
			# aux_model_res_2_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[6].output[1])
			
			# aux_model_bin_conv_3 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[7].output)
			# aux_model_res_3_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[9].output[0])
			# aux_model_res_3_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[9].output[1])
			
			# aux_model_res_4_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[13].output[0])
			# aux_model_res_4_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[13].output[1])
			
			# aux_model_res_5_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[17].output[0])
			# aux_model_res_5_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[17].output[1])
			
			# aux_model_res_6_bit_1 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[20].output[0])
			# aux_model_res_6_bit_2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[20].output[1])

			# aux_model_activation = tf.keras.Model(inputs=model.inputs, outputs=model.layers[23].output)

			# # Access intermediate outputs of the original model
			# bin_conv_1 = np.transpose(aux_model_bin_conv_1.predict(X_test), (0,3,1,2))
			# bnorm_1 = np.transpose(aux_model_bnorm_1.predict(X_test), (0,3,1,2))
			
			# residual_sign_1_bit_1 = np.transpose(aux_model_res_1_bit_1.predict(X_test), (0,3,1,2))
			# residual_sign_1_bit_2 = np.transpose(aux_model_res_1_bit_2.predict(X_test), (0,3,1,2))

			# bin_conv_2 = np.transpose(aux_model_bin_conv_2.predict(X_test), (0,3,1,2))			
			# residual_sign_2_bit_1 = np.transpose(aux_model_res_2_bit_1.predict(X_test), (0,3,1,2))
			# residual_sign_2_bit_2 = np.transpose(aux_model_res_2_bit_2.predict(X_test), (0,3,1,2))
						
			# bin_conv_3 = np.transpose(aux_model_bin_conv_3.predict(X_test), (0,3,1,2))			
			# residual_sign_3_bit_1 = np.transpose(aux_model_res_3_bit_1.predict(X_test), (0,3,1,2))
			# residual_sign_3_bit_2 = np.transpose(aux_model_res_3_bit_2.predict(X_test), (0,3,1,2))
						
			# residual_sign_4_bit_1 = np.transpose(aux_model_res_4_bit_1.predict(X_test), (0,3,1,2))
			# residual_sign_4_bit_2 = np.transpose(aux_model_res_4_bit_2.predict(X_test), (0,3,1,2))

			# residual_sign_5_bit_1 = aux_model_res_5_bit_1.predict(X_test)
			# residual_sign_5_bit_2 = aux_model_res_5_bit_2.predict(X_test)

			# residual_sign_6_bit_1 = aux_model_res_6_bit_1.predict(X_test)
			# residual_sign_6_bit_2 = aux_model_res_6_bit_2.predict(X_test)

			# acti = aux_model_activation.predict(X_test)


			# # Save intermediate outputs
			# for ch in range(16):
			# 	np.savetxt(output_path + '/bin_conv_1_ch_'+str(ch)+'.txt', bin_conv_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/bnorm_1_ch_'+str(ch)+'.txt', bnorm_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_1_bit_1_ch_'+str(ch)+'.txt', residual_sign_1_bit_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_1_bit_2_ch_'+str(ch)+'.txt', residual_sign_1_bit_2[0][ch][:][:])
			# 	np.savetxt(output_path + '/bin_conv_2_ch_'+str(ch)+'.txt', bin_conv_2[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_2_bit_1_ch_'+str(ch)+'.txt', residual_sign_2_bit_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_2_bit_2_ch_'+str(ch)+'.txt', residual_sign_2_bit_2[0][ch][:][:])

			# for ch in range(32):
			# 	np.savetxt(output_path + '/bin_conv_3_ch_'+str(ch)+'.txt', bin_conv_3[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_3_bit_1_ch_'+str(ch)+'.txt', residual_sign_3_bit_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_3_bit_2_ch_'+str(ch)+'.txt', residual_sign_3_bit_2[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_4_bit_1_ch_'+str(ch)+'.txt', residual_sign_4_bit_1[0][ch][:][:])
			# 	np.savetxt(output_path + '/residual_sign_4_bit_2_ch_'+str(ch)+'.txt', residual_sign_4_bit_2[0][ch][:][:])

			# np.savetxt(output_path + '/residual_sign_5_bit_1_ch_'+str(0)+'.txt', residual_sign_5_bit_1[0][:])
			# np.savetxt(output_path + '/residual_sign_5_bit_2_ch_'+str(0)+'.txt', residual_sign_5_bit_2[0][:])

			# np.savetxt(output_path + '/residual_sign_6_bit_1_ch_'+str(0)+'.txt', residual_sign_6_bit_1[0][:])
			# np.savetxt(output_path + '/residual_sign_6_bit_2_ch_'+str(0)+'.txt', residual_sign_6_bit_2[0][:])

			# np.savetxt(output_path + '/activation.txt', acti[0][:])


			logger.info("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))
			print("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))

		
		tf.Session().close()
		K.clear_session()

		return score[1]