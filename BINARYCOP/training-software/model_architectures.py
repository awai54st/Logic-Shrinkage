
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops
from binarization_utils import *

batch_norm_eps=1e-4
batch_norm_alpha=0.1

def get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed):


	if dataset=='MNIST':
		model=Sequential()
		model.add(binary_dense(levels=resid_levels,n_in=784,n_out=256,input_shape=[784],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(Activation('softmax'))
		

	elif dataset in ["CIFAR-10","SVHN"]:
		model=Sequential()
		model.add(binary_conv(nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_conv_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		model.add(binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))

		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))

		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5"))
		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='valid',LUT=LUT,REG=REG,k_lut=k_lut,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_6"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_6"))

		model.add(my_flat())

		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_7"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_7"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_8"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_8"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_9"))
		model.add(Activation('softmax'))
	
	elif dataset=="BinaryCoP":
		model=Sequential()
		
		# Layer Conv1_1 | [3, 64] in Binary CoP article Table 1
		# The images are resized to 32x32 pixels imilar to the CIFAR-10 dataset and as stated in Binary CoP article section IV.A
		model.add(binary_conv(nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_conv_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		
		# Layer Conv1_2 | [64, 64] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))


		# Layer Conv2_1 | [64, 128] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		
		# Layer Conv2_2 | [128, 128] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))
   
		
		# Layer Conv3_1 | [128, 256] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5"))

		# Layer Conv3_2 | [256, 256] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='valid',LUT=LUT,REG=REG,k_lut=k_lut,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_6"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_6"))
		model.add(my_flat())
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6"))


		# Layer FC1 | [512] in Binary CoP article Table 1
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[resid_levels]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_7"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_7"))
		
		# Layer FC2 | [512] in Binary CoP article Table 1
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[resid_levels]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_8"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_8"))

		# Layer FC3 | [4] in Binary CoP article Table 1
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[resid_levels]),n_out=4,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_9"))

		model.add(Activation('softmax'))

	elif dataset=="mu-BinaryCoP":
		# µ-CNV network architecture
		model=Sequential()
		
		# Layer Conv1_1 | [3, 16] in Binary CoP article Table 1
		# The images are resized to 32x32 pixels imilar to the CIFAR-10 dataset and as stated in Binary CoP article section IV.A
		model.add(binary_conv(nfilters=16,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_conv_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		
		# Layer Conv1_2 | [16, 16] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=16,ch_in=16,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))


		# Layer Conv2_1 | [16, 32] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=32,ch_in=16,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		
		# Layer Conv2_2 | [32, 32] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=32,ch_in=32,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))
   
		
		# Layer Conv3_1 | [32, 64] in Binary CoP article Table 1
		model.add(binary_conv(levels=resid_levels,nfilters=64,ch_in=32,k=3,padding='valid',LUT=LUT,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(my_flat())
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5"))


		# Layer FC1 | [128] in Binary CoP article Table 1
		model.add(binary_dense(levels=resid_levels,n_in=576,n_out=128,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_6"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6"))

		# Layer FC2 | [4] in Binary CoP article Table 1
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[resid_levels]),n_out=4,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_alpha, epsilon=batch_norm_eps, name="batch_normalization_7"))

		model.add(Activation('softmax'))

	else:
		raise("dataset should be one of the following: [MNIST, CIFAR-10, SVHN, BinaryCoP, mu-BinaryCoP].")

	return model
