
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization, ReLU, Input
from tensorflow.python.framework import ops
from binarization_utils import *
from binarization_utils_birealnet import *
#import random

batch_norm_eps=1e-4
batch_norm_momentum=0.9#(this is same as momentum)

#pruning_percentage = {
#  "binary_dense_1": -1,
#  "binary_dense_2": 0.9,
#  "binary_dense_3": 0.9,
#  "binary_dense_4": -1,
#}

#K = 3
#num_act_pruned = 1

def get_model(dataset,resid_levels,k_lut,LUT,REG,BINARY,LOGIC_SHRINKAGE,trainable_means,custom_rand_seed):

	#np.random.seed(rand_seed)
	#tf.set_random_seed(rand_seed)
	#random.seed(rand_seed)

	if dataset=='MNIST':
		model=Sequential()
		model.add(binary_dense(levels=resid_levels,n_in=784,n_out=256,input_shape=[784],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=256,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,k_lut=k_lut,LUT=LUT,REG=REG,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(Activation('softmax'))

	elif dataset=="CIFAR-10" or dataset=="SVHN":

		model=Sequential()
		model.add(binary_conv(nfilters=64,ch_in=3,k=3,padding='valid',input_shape=[32,32,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed, name="binary_conv_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_1"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))
		model.add(binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_2"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))

		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=64,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_3"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))
		model.add(binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_4"))
		model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))

		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=128,k=3,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_5"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5"))
		model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='valid',LUT=LUT,REG=REG,k_lut=k_lut,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_6"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_6"))

		model.add(my_flat())

		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_7"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_7"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=512,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_8"))
		model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_8"))
		model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=10,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_9"))
		model.add(Activation('softmax'))

	elif dataset == "IMAGENET":

		#model=Sequential()

		#model.add(binary_conv(levels=resid_levels,nfilters=64,ch_in=3,k=11,strides=(4,4),padding='valid',input_shape=[224,224,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_1"))
		#model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_1"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1"))

		#model.add(binary_conv(levels=resid_levels,nfilters=192,ch_in=64,k=5,padding='valid',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2"))
		#model.add(MaxPooling2D(pool_size=(3, 3),strides=(2,2)))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_2"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2"))


		#model.add(binary_conv(levels=resid_levels,nfilters=384,ch_in=192,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_3"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3"))

		#model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=384,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_4"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4"))

		#model.add(binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_5"))

		#model.add(my_flat())

		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5"))

		#model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=4096,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_6"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6"))
		##model.add(Dropout(0.5))
		#model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=4096,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_2"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_7"))
		#model.add(Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_7"))
		##model.add(Dropout(0.5))

		#model.add(binary_dense(levels=resid_levels,n_in=int(model.output.get_shape()[2]),n_out=1000,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_3"))
		#model.add(BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_8"))
		##model.add(Dropout(0.5))

		#model.add(Activation('softmax'))






		input_layer = Input(shape=[224,224,3])
		# conv1
		x = birealnet_binary_conv(levels=resid_levels,nfilters=64,ch_in=3,k=7,strides=(2,2),padding='same',input_shape=[224,224,3],first_layer=True,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_1")(input_layer)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_1")(x)
		x = MaxPooling2D(pool_size=(3, 3),strides=(2,2))(x)
		
		# conv2_x
		shortcut = x
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_1")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_2")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_2")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_2")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_3")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_3")(x)
		x = x + shortcut
		shortcut = x
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_3")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_4")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_4")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_4")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=64,ch_in=64,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_5")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_5")(x)
		x = x + shortcut
		
		# conv3_x
		shortcut = Conv2D(128, 1, strides=(2, 2), padding="same")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_5")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=128,ch_in=64,k=3,padding='same',strides=(2,2),LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_6")(x)
		#x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_6")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_6")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_7")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_7")(x)
		x = x + shortcut
		shortcut = x
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_7")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_8")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_8")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_8")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=128,ch_in=128,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_9")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_9")(x)
		x = x + shortcut

		# conv4_x
		shortcut = Conv2D(256, 1, strides=(2, 2), padding="same")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_9")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=256,ch_in=128,k=3,padding='same', strides=(2, 2),LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_10")(x)
		#x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_10")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_10")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_11")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_11")(x)
		x = x + shortcut
		shortcut = x
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_11")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='same',LUT=LUT,REG=REG,k_lut=k_lut,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_12")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_12")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_12")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=256,ch_in=256,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_13")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_13")(x)
		x = x + shortcut
		
		# conv5_x
		shortcut = Conv2D(512, 1, strides=(2, 2), padding="same")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_13")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=512,ch_in=256,k=3,padding='same', strides=(2, 2),LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_14")(x)
		#x = MaxPooling2D(pool_size=(2, 2),strides=(2,2))(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_14")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_14")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=512,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_15")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_15")(x)
		x = x + shortcut
		shortcut = x
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_15")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=512,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_16")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_16")(x)
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_16")(x)
		x = birealnet_binary_conv(levels=resid_levels,nfilters=512,ch_in=512,k=3,padding='same',LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_conv_17")(x)
		x = BatchNormalization(axis=-1, momentum=batch_norm_momentum, epsilon=batch_norm_eps, name="batch_normalization_17")(x)
		x = x + shortcut

		# Logit
		x = tf.reduce_mean(x, [1, 2])
		x = BiRealNet_Residual_sign(levels=resid_levels,trainable=trainable_means, name="residual_sign_17")(x)
		x = binary_dense(levels=resid_levels,n_in=512,n_out=1000,LUT=False,BINARY=BINARY,LOGIC_SHRINKAGE=LOGIC_SHRINKAGE,custom_rand_seed=custom_rand_seed,name="binary_dense_1")(x)
		x = Activation('softmax')(x)
		model = Model(inputs=input_layer, outputs=x)


	return model
