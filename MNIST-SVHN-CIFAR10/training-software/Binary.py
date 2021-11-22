import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
from tensorflow.keras.datasets import cifar10,mnist
import tensorflow.keras.utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
from binarization_utils import *

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


def train_lutnet(get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, epochs, batch_size, output_path='models', custom_rand_seed=0):

	print('Train is ', Train)
	print('REG is ', REG)
	print('Retrain is ', Retrain)
	print('LUT is ', LUT)
	print('BINARY is ', BINARY)
	print('LOGIC_SHRINKAGE is ', LOGIC_SHRINKAGE)
	print('trainable_means is ', trainable_means)
	print('Evaluate is ', Evaluate)

	dataset = "CIFAR-10"
	
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
	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN].")
	
	X_train=X_train.astype(np.float32)
	X_test=X_test.astype(np.float32)
	Y_train = to_categorical(y_train, 10)
	Y_test = to_categorical(y_test, 10)
	X_train /= 255
	X_test /= 255
	X_train=2*X_train-1
	X_test=2*X_test-1
	
	
	print('X_train shape:', X_train.shape)
	print(X_train.shape[0], 'train samples')
	print(X_test.shape[0], 'test samples')
	
	if Train:
		if not(os.path.exists(output_path)):
			os.mkdir(output_path)
		for resid_levels in range(2,3): #range(1,4):
			print ('training with', resid_levels,'levels')
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
				if dataset=="CIFAR-10":
					horizontal_flip=True
				if dataset=="SVHN":
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
			model=get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed)
			model.load_weights(weights_path)
			opt = keras.optimizers.Adam()
			model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
			#model.summary()
			score=model.evaluate(X_test,Y_test,verbose=0)
			print ("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))

		tf.Session().close()
		K.clear_session()

		return score[1]

