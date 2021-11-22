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
#import random
# To show training progress
from tqdm.keras import TqdmCallback

# Using keras' built-in mobilenet preprocessing method
from keras.applications import mobilenet
from keras.applications.mobilenet import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import tensorflow_datasets as tfds
import cv2 as cv

from tensorflow.keras import layers
from imagenet_preproc_util import *

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

def load_imagenet(path_to_dataset):
	imagenet_ds, info_ds = tfds.load('imagenet2012', 
		data_dir=path_to_dataset,         
		#split=['train', 'validation'],
		#batch_size=batch_size,
		shuffle_files=False, 
		download=False, 
		as_supervised=True,
		with_info=True)
	return imagenet_ds, info_ds
	
def resize_with_crop(image, label):
	i = image
	i = tf.cast(i, tf.float32)
	# Normalize
	i = i / 255 * 2 - 1
	i = tf.image.resize_with_crop_or_pad(i, 224, 224)
	i = tf.keras.applications.mobilenet_v2.preprocess_input(i)
	return (i, label)

def imagenet_resize_image(input_image, random_aspect=False):
	# Resize image so that the shorter side is 256
	height_orig = tf.shape(input_image)[0]
	width_orig = tf.shape(input_image)[1]
	ratio_flag = tf.greater(height_orig, width_orig)  # True if height > width

	aspect_ratio = tf.random.uniform([], minval=0.875, maxval=1.2, dtype=tf.float64)
	height = tf.where(ratio_flag, tf.cast(256*height_orig/width_orig*aspect_ratio, tf.int32), 256)
	width = tf.where(ratio_flag, 256, tf.cast(256*width_orig/height_orig*aspect_ratio, tf.int32))

	image = tf.image.resize(input_image, [height, width])
	return image


def imagenet_preprocess(image):
	# Resize_image
	image = imagenet_resize_image(image)
	
	# Crop(random/center)
	height = 224
	width = 224
	image = tf.image.random_crop(image, [height, width, 3])
	
	# Preprocess: imagr normalization per channel
	imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
	imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
	image = (image - imagenet_mean) / imagenet_std
	
	return image

def imagenet_generator(dataset, batch_size=32, num_classes=1000, is_training=False):
	images = np.zeros((batch_size, 224, 224, 3))
	labels = np.zeros((batch_size, num_classes))
	while True:
		count = 0 
		for sample in tfds.as_numpy(dataset):
			#print(np.shape(sample[0]))
			#print(np.shape(sample[1]))
			image = sample[0]
			label = sample[1]
    
			images[count%batch_size] = mobilenet.preprocess_input(np.expand_dims(image, 0))
			labels[count%batch_size] = np.expand_dims(to_categorical(label, num_classes=num_classes), 0)
      
			count += 1
			if (count%batch_size == 0):
				yield images, labels

def imagenet_training_dataset_preprocessing(dataset):
	lighting_param = 0.1
	augmentation_training = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(256, 256),
		layers.experimental.preprocessing.RandomCrop(224, 224),
		Lighting(lighting_param),
		layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
		layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])
	])
	dataset_aug = dataset.map(
		lambda x, y: (augmentation_training(x, training=True), y))
	return dataset_aug

def prepare_training_dataset(dataset, batch_size):
	# Resize and crop dataset
	resize = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(256, 256),
	#	layers.experimental.preprocessing.RandomCrop(224, 224),
	])
	dataset = dataset.map(lambda x, y: (resize(x), y), 
		num_parallel_calls=tf.data.AUTOTUNE)
	# Shuffle dataset
	dataset = dataset.shuffle(1000)
	# Batch all datasets
	dataset = dataset.batch(batch_size)
	# Dataset augmentation
	lighting_param = 0.1
	augmentation = tf.keras.Sequential([
		layers.experimental.preprocessing.RandomCrop(224, 224),
		Lighting(lighting_param),
		layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
		layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])
	])
	dataset = dataset.map(
		lambda x, y: (augmentation(x, training=True), y))
	return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def imagenet_validation_dataset_preprocessing(dataset):
	augmentation_validation = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(256, 256),
		layers.experimental.preprocessing.CenterCrop(224, 224),
		layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])
	])
	dataset_aug = dataset.map(
		lambda x, y: (augmentation_validation(x), y))
	return dataset_aug

def prepare_validation_dataset(dataset, batch_size):
	# Resize and crop dataset
	resize = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(256, 256),
	#	layers.experimental.preprocessing.CenterCrop(224, 224),
	])
	dataset = dataset.map(lambda x, y: (resize(x), y), 
		num_parallel_calls=tf.data.AUTOTUNE)
	# Batch all datasets
	dataset = dataset.batch(batch_size)
	# Dataset augmentation
	augmentation = tf.keras.Sequential([
		layers.experimental.preprocessing.Resizing(256, 256),
		layers.experimental.preprocessing.CenterCrop(224, 224),
		layers.experimental.preprocessing.Normalization(mean=[0.485, 0.456, 0.406], variance=[0.229 ** 2, 0.224 ** 2, 0.225 ** 2])
	])
	dataset = dataset.map(
		lambda x, y: (augmentation(x, training=False), y))
	return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


def train_lutnet(get_model, pruning_threshold, k_lut, Train, REG, Retrain, LUT, BINARY, LOGIC_SHRINKAGE, trainable_means, Evaluate, epochs, batch_size, output_path='models', custom_rand_seed=0):

	#np.random.seed(custom_rand_seed)
	#random.seed(rand_seed)

	print('Train is ', Train)
	print('REG is ', REG)
	print('Retrain is ', Retrain)
	print('LUT is ', LUT)
	print('BINARY is ', BINARY)
	print('LOGIC_SHRINKAGE is ', LOGIC_SHRINKAGE)
	print('trainable_means is ', trainable_means)
	print('Evaluate is ', Evaluate)

	dataset = "IMAGENET"
	
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
	elif dataset=="IMAGENET":
		use_generator=False
		#(X_train, y_train), (X_test, y_test) = load_imagenet('/home/storage/imagenet_original')
		imagenet_ds, info_ds = load_imagenet('/home/LUTNet-NAS/unrolled-lutnet/training-software/datasets/tf-imagenet-dirs/data')
		# Preprocess the images
		#train_ds = train_ds.map(resize_with_crop)
		#test_ds = test_ds.map(resize_with_crop)
		#train_ds = train_ds.map(imagenet_preprocess).batch(batch_size)
		#test_ds = test_ds.map(imagenet_preprocess).batch(batch_size)
		train_dataset, validation_dataset = imagenet_ds['train'], imagenet_ds['validation']
		assert isinstance(train_dataset, tf.data.Dataset)
		assert isinstance(validation_dataset, tf.data.Dataset)
		train_dataset = prepare_training_dataset(train_dataset, batch_size)
		validation_dataset = prepare_validation_dataset(validation_dataset, batch_size)
		## Batching
		#train_dataset = train_dataset.batch(batch_size)
		#validation_dataset = validation_dataset.batch(batch_size)
		## Augmentation
		#train_dataset = imagenet_training_dataset_preprocessing(train_dataset)
		#validation_dataset = imagenet_validation_dataset_preprocessing(validation_dataset)
		## Buffered prefetching
		#train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
		#validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
		# Get dataset size
		train_size = info_ds.splits['train'].num_examples
		validation_size = info_ds.splits['validation'].num_examples
	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, IMAGENET].")

	if dataset!="IMAGENET":
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

		resid_levels = 2 # Number of bits for inputs set to 2
		print ('training with', resid_levels,'levels')

		#sess=K.get_session()
		sess=tf.compat.v1.keras.backend.get_session()

		# Create a MirroredStrategy for multiple GPU training.
		#strategy = tf.distribute.MirroredStrategy()
		mirrored_strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
		print("Number of devices: {}".format(mirrored_strategy.num_replicas_in_sync))

		# Open a strategy scope.
		with mirrored_strategy.scope():

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
	
			weights_path = output_path + '/'+str(resid_levels)+'_residuals.h5'
			if Retrain: #pruned retraining
				#weights_path='models/pretrained_pruned.h5'
				model.load_weights(weights_path)
			elif LUT and not REG: #logic expansion
				#weights_path='models/pretrained_bin.h5'
				model.load_weights(weights_path)

			if dataset == "IMAGENET":
				lr=0.0005
				opt = keras.optimizers.Adam(lr=lr)
			else:
				lr=0.01
				decay=1e-6
				opt = keras.optimizers.Adam(lr=lr, decay=decay)

			#opt = keras.optimizers.SGD(lr=lr,momentum=0.9)
			if dataset == "IMAGENET":
				#model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
				model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
			else:
				model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
	
			# Learning rate scheduler (for ImageNet only)
			#def scheduler(epoch, lr):
			#	if epoch < 10:
			#		return lr
			#	elif epoch < 15:
			#		return lr * 0.1
			#	else:
			#		return lr * 0.01
			def scheduler(epoch, lr): # A linear lr decay schedula for ImageNet
				initial_lr = 0.0005
				num_epochs = 64
				return lr - initial_lr/num_epochs

			cback=keras.callbacks.ModelCheckpoint(weights_path, monitor='val_accuracy', save_best_only=True)
			if use_generator:
				if dataset=="CIFAR-10":
					horizontal_flip=True
				if dataset=="SVHN":
					horizontal_flip=False
				datagen = ImageDataGenerator(
					width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
					height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
					horizontal_flip=horizontal_flip)  # randomly flip images
				history=model.fit_generator(datagen.flow(X_train, y_train,batch_size=batch_size),steps_per_epoch=X_train.shape[0]//batch_size,
				epochs=epochs,validation_data=(X_test, y_test),verbose=2,callbacks=[cback])
	
			else:
				if dataset == "IMAGENET":
					lr_cback = keras.callbacks.LearningRateScheduler(scheduler)
					#history=model.fit(train_ds,validation_data=test_ds, verbose=0,epochs=epochs,steps_per_epoch=train_size//batch_size,validation_steps=test_size//batch_size,callbacks=[cback, lr_cback, TqdmCallback(verbose=2)])
					#history=model.fit(train_ds,validation_data=test_ds, verbose=2,epochs=epochs,steps_per_epoch=train_size//batch_size,validation_steps=test_size//batch_size,callbacks=[cback, lr_cback])
					#model.fit_generator(imagenet_generator(train_dataset, batch_size=batch_size, is_training=True),
					#	steps_per_epoch= train_size // batch_size,
					#	epochs = epochs,
					#	validation_data = imagenet_generator(validation_dataset, batch_size=batch_size),
					#	validation_steps = validation_size // batch_size,
					#	verbose = 0,
					#	callbacks = [cback, lr_cback, TqdmCallback(verbose=2)])
					model.fit(train_dataset,
						#steps_per_epoch= train_size // batch_size,
						epochs = epochs,
						validation_data = validation_dataset,
						#validation_steps = validation_size // batch_size,
						verbose = 2,
						callbacks = [cback, lr_cback])

				else:
					history=model.fit(X_train, y_train,batch_size=batch_size,validation_data=(X_test, y_test), verbose=2,epochs=epochs,callbacks=[cback])
			#dic={'hard':history.history}
			#foo=open(output_path + '/history_'+str(resid_levels)+'_residuals.pkl','wb')
			#pickle.dump(dic,foo)
			#foo.close()
		
		#tf.Session().close()
		tf.compat.v1.Session().close()
		K.clear_session()

	if Evaluate:
		resid_levels = 2 # Number of bits for inputs set to 2

		weights_path = output_path + '/'+str(resid_levels)+'_residuals.h5'
		model=get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed)
		model.load_weights(weights_path)
		opt = keras.optimizers.Adam()
		model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
		#model.summary()
		if dataset == "IMAGENET":
			score=model.evaluate(validation_dataset,verbose=0)
		else:
			score=model.evaluate(X_test,Y_test,verbose=0)
		print ("with %d residuals, test loss was %0.4f, test accuracy was %0.4f"%(resid_levels,score[0],score[1]))

		#tf.Session().close()
		tf.compat.v1.Session().close()
		K.clear_session()

		return score[1]

