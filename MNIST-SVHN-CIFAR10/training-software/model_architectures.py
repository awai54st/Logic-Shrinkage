
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
		

	elif dataset=="CIFAR-10" or dataset=="SVHN":

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


	return model
