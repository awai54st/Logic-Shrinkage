import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Convolution2D, Activation, Flatten, MaxPooling2D,Input,Dropout,GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10
import tensorflow.keras.utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.framework import ops
#import random
from binarization_utils import genkernel_tf, genkernel_mat_tf, dummy_kernel_mat

def BiRealNetBinaryActivation(x):
    out_forward = K.sign(x)
    #out_e1 = (x^2 + 2*x)
    #out_e2 = (-x^2 + 2*x)
    out_e_total = 0
    mask1 = tf.dtypes.cast(x < -1, tf.float32)
    mask2 = tf.dtypes.cast(x < 0, tf.float32)
    mask3 = tf.dtypes.cast(x < 1, tf.float32)
    out1 = (-1) * mask1 + (x*x + 2*x) * (1-mask1)
    out2 = out1 * mask2 + (-x*x + 2*x) * (1-mask2)
    out3 = out2 * mask3 + 1 * (1- mask3)
    out = out3 + K.stop_gradient(out_forward - out3)

    return out

def BiRealNetMagnitudeAwareWeight(w, num_dims=4):
    real_weights = w
    #scaling_factor = K.mean(K.mean(K.mean(abs(real_weights),axis=0,keepdim=True),axis=1,keepdim=True),axis=2,keepdim=True)
    if num_dims == 4:
        scaling_factor = tf.reduce_mean(abs(real_weights),axis=[0,1,2],keepdims=True)
    elif num_dims == 5:
        scaling_factor = tf.reduce_mean(abs(real_weights),axis=[1,2,3],keepdims=True)

    #print(scaling_factor, flush=True)
    scaling_factor = K.stop_gradient(scaling_factor)
    binary_weights_no_grad = scaling_factor * K.sign(real_weights)
    cliped_weights = K.clip(real_weights, -1.0, 1.0)
    binary_weights = K.stop_gradient(binary_weights_no_grad) - K.stop_gradient(cliped_weights) + cliped_weights

    return binary_weights

class BiRealNet_Residual_sign(Layer):
    def __init__(self, levels=1,trainable=True,**kwargs):
        self.levels=levels
        self.trainable=trainable
        super(BiRealNet_Residual_sign, self).__init__(**kwargs)
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'levels': self.levels,
            'trainable': self.trainable,
        })
        return config

    def build(self, input_shape):
        ars=np.arange(self.levels)+1.0
        ars=ars[::-1]
        means=ars/np.sum(ars)
        #self.means=[K.variable(m) for m in means]
        #self.trainable_weights=self.means
        self.means = self.add_weight(name='means',
            shape=(self.levels, ),
            initializer=keras.initializers.Constant(value=means),
            trainable=self.trainable) # Trainable scaling factors for residual binarisation
    def call(self, x, mask=None):
        resid = x
        out_bin=0

        if self.levels==1:
            for l in range(self.levels):
                #out=binarize(resid)*K.abs(self.means[l])
                out=BiRealNetBinaryActivation(resid)*abs(self.means[l])
                #out_bin=out_bin+out
                out_bin=out_bin+out#no gamma per level
                resid=resid-out
        elif self.levels==2:
            out=BiRealNetBinaryActivation(resid)*abs(self.means[0])
            out_bin=out
            resid=resid-out
            out=BiRealNetBinaryActivation(resid)*abs(self.means[1])
            out_bin=tf.stack([out_bin,out])
            resid=resid-out
        elif self.levels==3:
            out=BiRealNetBinaryActivation(resid)*abs(self.means[0])
            out_bin1=out
            resid=resid-out
            out=BiRealNetBinaryActivation(resid)*abs(self.means[1])
            out_bin2=out
            resid=resid-out
            out=BiRealNetBinaryActivation(resid)*abs(self.means[2])
            out_bin3=out
            resid=resid-out
            out_bin=tf.stack([out_bin1,out_bin2,out_bin3])

                
        return out_bin

    def get_output_shape_for(self,input_shape):
        if self.levels==1:
            return input_shape
        else:
            return (self.levels, input_shape)
    def compute_output_shape(self,input_shape):
        if self.levels==1:
            return input_shape
        else:
            return (self.levels, input_shape)
    def set_means(self,X):
        means=np.zeros((self.levels))
        means[0]=1
        resid=np.clip(X,-1,1)
        approx=0
        for l in range(self.levels):
            m=np.mean(np.absolute(resid))
            out=np.sign(resid)*m
            approx=approx+out
            resid=resid-out
            means[l]=m
            err=np.mean((approx-np.clip(X,-1,1))**2)

        means=means/np.sum(means)
        sess=K.get_session()
        sess.run(self.means.assign(means))

class birealnet_binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,strides=(1,1),levels=1,k_lut=4,first_layer=False,LUT=True,REG=None,BINARY=True,LOGIC_SHRINKAGE=False,custom_rand_seed=0,**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		if padding=='valid':
			self.PADDING = "VALID" #tf uses upper-case padding notations whereas keras uses lower-case notations
		elif padding=='same':
			self.PADDING = "SAME"
		self.strides=strides
		self.levels=levels
		self.k_lut=k_lut
		self.first_layer=first_layer
		self.LUT=LUT
		self.REG=REG
		self.BINARY=BINARY
		self.LOGIC_SHRINKAGE=LOGIC_SHRINKAGE
		#self.custom_rand_seed=custom_rand_seed
		self.window_size=self.ch_in*self.k*self.k
		super(birealnet_binary_conv,self).__init__(**kwargs)
	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'nfilters': self.nfilters,
			'ch_in': self.ch_in,
			'k': self.k,
			'padding': self.padding,
			'strides': self.strides,
			'levels': self.levels,
			'k_lut': self.k_lut,
			'first_layer': self.first_layer,
			'LUT': self.LUT,
			'REG': self.REG,
			'BINARY': self.BINARY,
			'LOGIC_SHRINKAGE': self.LOGIC_SHRINKAGE,
			#'custom_rand_seed': self.custom_rand_seed,
		})
		return config

	def build(self, input_shape):

		#np.random.seed(self.custom_rand_seed)
		#tf.set_random_seed(self.rand_seed)
		#random.seed(self.rand_seed)

		if self.k_lut > 1:
			self.rand_map_0 = self.add_weight(name='rand_map_0', 
				shape=(self.window_size, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 2:
			self.rand_map_1 = self.add_weight(name='rand_map_1', 
				shape=(self.window_size, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 3:
			self.rand_map_2 = self.add_weight(name='rand_map_2', 
				shape=(self.window_size, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 4:
			self.rand_map_3 = self.add_weight(name='rand_map_3', 
				shape=(self.window_size, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 5:
			self.rand_map_4 = self.add_weight(name='rand_map_4', 
				shape=(self.window_size, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.window_size, size=[self.window_size, 1])),
				trainable=False) # Randomisation map for subsequent input connections

		stdv=1/np.sqrt(self.k*self.k*self.ch_in)

		if self.levels==1 or self.first_layer==True:
			#w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
			#self.w=K.variable(w)
			#self._trainable_weights=[self.w,self.gamma]
			if self.REG==True:
				self.w = self.add_weight(
					name='Variable_1',
					shape=[self.k,self.k,self.ch_in,self.nfilters],
					initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
					regularizer=tf.keras.regularizers.l2(5e-7),
					trainable=True
				)
			else:
				self.w = self.add_weight(
					name='Variable_1',
					shape=[self.k,self.k,self.ch_in,self.nfilters],
					initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
					trainable=True
				)

		elif self.levels==2:
			if self.LUT==True:
				#c_param = np.random.normal(loc=0.0, scale=stdv,size=[(2**self.k_lut)*self.levels,self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				#self.c_param =K.variable(c_param)
				#self._trainable_weights=[self.c_param, self.gamma]

				if self.REG==True:
					self.c_param = self.add_weight(
						name='Variable_1',
						shape=[(2**self.k_lut)*self.levels,self.k,self.k,self.ch_in,self.nfilters],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						regularizer=tf.keras.regularizers.l2(5e-7),
						trainable=True
					)
				else:
					self.c_param = self.add_weight(
						name='Variable_1',
						shape=[(2**self.k_lut)*self.levels,self.k,self.k,self.ch_in,self.nfilters],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						trainable=True
					)


			else:
				#w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				#self.w=K.variable(w)
				#self._trainable_weights=[self.w,self.gamma]

				if self.REG==True:
					self.w = self.add_weight(
						name='Variable_1',
						shape=[self.k,self.k,self.ch_in,self.nfilters],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						regularizer=tf.keras.regularizers.l2(5e-7),
						trainable=True
					)
				else:
					self.w = self.add_weight(
						name='Variable_1',
						shape=[self.k,self.k,self.ch_in,self.nfilters],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						trainable=True
					)


		elif self.levels==3:
			assert False

		if self.LUT == True: # Construct shrinkage maps

			value=dummy_kernel_mat(K=self.k_lut,num_ops=self.window_size*self.nfilters)
			if self.k_lut > 4: # Split the shrinkage matrix to avoid the 2GB tensor size limit
				value = np.split(value, 8, axis=-1)
				self.shrinkage_map = [None] * 8
				for i in range(8):
					self.shrinkage_map[i] = self.add_weight(name='shrinkage_map_p'+str(i),
						shape=value[i].shape,
						initializer=keras.initializers.Constant(value[i]),
						trainable=False)
				#self.shrinkage_map_p0 = self.add_weight(name='shrinkage_map_p0',
				#	shape=value[0].shape,
				#	initializer=keras.initializers.Constant(value[0]),
				#	trainable=False)
				#self.shrinkage_map_p1 = self.add_weight(name='shrinkage_map_p1',
				#	shape=value[1].shape,
				#	initializer=keras.initializers.Constant(value[1]),
				#	trainable=False)
				#self.shrinkage_map_p2 = self.add_weight(name='shrinkage_map_p2',
				#	shape=value[2].shape,
				#	initializer=keras.initializers.Constant(value[2]),
				#	trainable=False)
				#self.shrinkage_map_p3 = self.add_weight(name='shrinkage_map_p3',
				#	shape=value[3].shape,
				#	initializer=keras.initializers.Constant(value[3]),
				#	trainable=False)
				#self.shrinkage_map_p4 = self.add_weight(name='shrinkage_map_p4',
				#	shape=value[4].shape,
				#	initializer=keras.initializers.Constant(value[4]),
				#	trainable=False)
				#self.shrinkage_map_p5 = self.add_weight(name='shrinkage_map_p5',
				#	shape=value[5].shape,
				#	initializer=keras.initializers.Constant(value[5]),
				#	trainable=False)
				#self.shrinkage_map_p6 = self.add_weight(name='shrinkage_map_p6',
				#	shape=value[6].shape,
				#	initializer=keras.initializers.Constant(value[6]),
				#	trainable=False)
				#self.shrinkage_map_p7 = self.add_weight(name='shrinkage_map_p7',
				#	shape=value[7].shape,
				#	initializer=keras.initializers.Constant(value[7]),
				#	trainable=False)
	
			else:
				self.shrinkage_map = self.add_weight(name='shrinkage_map',
					shape=value.shape,
					initializer=keras.initializers.Constant(value),
					trainable=False)

		self.pruning_mask = self.add_weight(name='pruning_mask',
			shape=(self.window_size,self.nfilters),
			initializer=keras.initializers.Constant(value=np.ones((self.window_size,self.nfilters))),
			trainable=False) # LUT pruning based on whether inputs get repeated


	def call(self, x,mask=None):

		#np.random.seed(self.custom_rand_seed)
		#tf.set_random_seed(self.rand_seed)
		#random.seed(self.rand_seed)

		if self.levels==1 or self.first_layer==True:
			if self.BINARY==False:
				self.clamped_w=K.clip(self.w,-1,1)
			else:
				self.clamped_w=BiRealNetMagnitudeAwareWeight(self.w, num_dims=4)
		elif self.levels==2:
			if self.LUT==True:

				if self.LOGIC_SHRINKAGE==True:

					#SHRINKAGE
					#Stacking c coefficients to create c_mat1 and c_mat2 tensors

					c_mat1=self.c_param[0 : 2**self.k_lut, :, :]
					c_mat2=self.c_param[2**self.k_lut : (2**self.k_lut)*2, :, :]
					
					c_mat1=tf.reshape(c_mat1,(1,2**self.k_lut,self.window_size*self.nfilters))
					c_mat2=tf.reshape(c_mat2,(1,2**self.k_lut,self.window_size*self.nfilters))
					
					c_mat1=tf.transpose(c_mat1,(2,0,1))
					c_mat2=tf.transpose(c_mat2,(2,0,1))

					if self.k_lut > 4:
						self.shrinkage_map = tf.concat(self.shrinkage_map, axis=2)

					c_mat1=tf.matmul(c_mat1,self.shrinkage_map)
					c_mat2=tf.matmul(c_mat2,self.shrinkage_map)
					
					c_mat1=tf.transpose(c_mat1,(2,0,1))
					c_mat2=tf.transpose(c_mat2,(2,0,1))
					
					c_mat1 = tf.reshape(c_mat1, (2**self.k_lut, self.k, self.k, self.ch_in, self.nfilters))
					c_mat2 = tf.reshape(c_mat2, (2**self.k_lut, self.k, self.k, self.ch_in, self.nfilters))

				
  
				if self.BINARY==False:

					self.clamped_c_param =K.clip(self.c_param,  -1,1)

				elif self.LOGIC_SHRINKAGE==True:

					self.clamped_c_param= BiRealNetMagnitudeAwareWeight(tf.concat([c_mat1, c_mat2], axis=0), num_dims=5)

				else:

					self.clamped_c_param =BiRealNetMagnitudeAwareWeight(self.c_param, num_dims=5)

			else:
				if self.BINARY==False:
					self.clamped_w=K.clip(self.w,-1,1)
				else:
					self.clamped_w=BiRealNetMagnitudeAwareWeight(self.w, num_dims=4)
		elif self.levels==3:
			assert False # 3-bit residual binarisation not implemented

		if keras.__version__[0]=='2':

			if self.levels==1 or self.first_layer==True:
				self.out=K.conv2d(x, kernel=self.clamped_w*tf.reshape(self.pruning_mask, [self.k, self.k, self.ch_in, self.nfilters]), padding=self.padding,strides=self.strides )
			elif self.levels==2:
				if self.LUT==True:
					x0_patches = tf.image.extract_patches(x[0,:,:,:,:],
						[1, self.k, self.k, 1],
						[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
						padding=self.PADDING)
					x1_patches = tf.image.extract_patches(x[1,:,:,:,:],
						[1, self.k, self.k, 1],
						[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
						padding=self.PADDING)

					# Special hack for randomising the subsequent input connections: tensorflow does not support advanced matrix indexing
					x0_shuf_patches=tf.transpose(x0_patches, perm=[3, 0, 1, 2])
					if self.k_lut > 1:
						x0_shuf_patches_0 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_0, tf.int32))
						x0_shuf_patches_0=tf.transpose(x0_shuf_patches_0, perm=[1, 2, 3, 0])
					if self.k_lut > 2:
						x0_shuf_patches_1 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_1, tf.int32))
						x0_shuf_patches_1=tf.transpose(x0_shuf_patches_1, perm=[1, 2, 3, 0])
					if self.k_lut > 3:
						x0_shuf_patches_2 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_2, tf.int32))
						x0_shuf_patches_2=tf.transpose(x0_shuf_patches_2, perm=[1, 2, 3, 0])
					if self.k_lut > 4:
						x0_shuf_patches_3 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_3, tf.int32))
						x0_shuf_patches_3=tf.transpose(x0_shuf_patches_3, perm=[1, 2, 3, 0])
					if self.k_lut > 5:
						x0_shuf_patches_4 = tf.gather_nd(x0_shuf_patches, tf.cast(self.rand_map_4, tf.int32))
						x0_shuf_patches_4=tf.transpose(x0_shuf_patches_4, perm=[1, 2, 3, 0])
		
					x1_shuf_patches=tf.transpose(x1_patches, perm=[3, 0, 1, 2])
					if self.k_lut > 1:
						x1_shuf_patches_0 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_0, tf.int32))
						x1_shuf_patches_0=tf.transpose(x1_shuf_patches_0, perm=[1, 2, 3, 0])
					if self.k_lut > 2:
						x1_shuf_patches_1 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_1, tf.int32))
						x1_shuf_patches_1=tf.transpose(x1_shuf_patches_1, perm=[1, 2, 3, 0])
					if self.k_lut > 3:
						x1_shuf_patches_2 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_2, tf.int32))
						x1_shuf_patches_2=tf.transpose(x1_shuf_patches_2, perm=[1, 2, 3, 0])
					if self.k_lut > 4:
						x1_shuf_patches_3 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_3, tf.int32))
						x1_shuf_patches_3=tf.transpose(x1_shuf_patches_3, perm=[1, 2, 3, 0])
					if self.k_lut > 5:
						x1_shuf_patches_4 = tf.gather_nd(x1_shuf_patches, tf.cast(self.rand_map_4, tf.int32))
						x1_shuf_patches_4=tf.transpose(x1_shuf_patches_4, perm=[1, 2, 3, 0])
	
					x0_pos=(1+BiRealNetBinaryActivation(x0_patches))/2*abs(x0_patches)
					x0_neg=(1-BiRealNetBinaryActivation(x0_patches))/2*abs(x0_patches)
					x1_pos=(1+BiRealNetBinaryActivation(x1_patches))/2*abs(x1_patches)
					x1_neg=(1-BiRealNetBinaryActivation(x1_patches))/2*abs(x1_patches)
					if self.k_lut > 1:
						x0s0_pos=(1+BiRealNetBinaryActivation(x0_shuf_patches_0))/2#*abs(x0_shuf_patches_0)
						x0s0_neg=(1-BiRealNetBinaryActivation(x0_shuf_patches_0))/2#*abs(x0_shuf_patches_0)
						x1s0_pos=(1+BiRealNetBinaryActivation(x1_shuf_patches_0))/2#*abs(x1_shuf_patches_0)
						x1s0_neg=(1-BiRealNetBinaryActivation(x1_shuf_patches_0))/2#*abs(x1_shuf_patches_0)
					if self.k_lut > 2:
						x0s1_pos=(1+BiRealNetBinaryActivation(x0_shuf_patches_1))/2#*abs(x0_shuf_patches_1)
						x0s1_neg=(1-BiRealNetBinaryActivation(x0_shuf_patches_1))/2#*abs(x0_shuf_patches_1)
						x1s1_pos=(1+BiRealNetBinaryActivation(x1_shuf_patches_1))/2#*abs(x1_shuf_patches_1)
						x1s1_neg=(1-BiRealNetBinaryActivation(x1_shuf_patches_1))/2#*abs(x1_shuf_patches_1)
					if self.k_lut > 3:
						x0s2_pos=(1+BiRealNetBinaryActivation(x0_shuf_patches_2))/2#*abs(x0_shuf_patches_2)
						x0s2_neg=(1-BiRealNetBinaryActivation(x0_shuf_patches_2))/2#*abs(x0_shuf_patches_2)
						x1s2_pos=(1+BiRealNetBinaryActivation(x1_shuf_patches_2))/2#*abs(x1_shuf_patches_2)
						x1s2_neg=(1-BiRealNetBinaryActivation(x1_shuf_patches_2))/2#*abs(x1_shuf_patches_2)
					if self.k_lut > 4:
						x0s3_pos=(1+BiRealNetBinaryActivation(x0_shuf_patches_3))/2#*abs(x0_shuf_patches_2)
						x0s3_neg=(1-BiRealNetBinaryActivation(x0_shuf_patches_3))/2#*abs(x0_shuf_patches_2)
						x1s3_pos=(1+BiRealNetBinaryActivation(x1_shuf_patches_3))/2#*abs(x1_shuf_patches_2)
						x1s3_neg=(1-BiRealNetBinaryActivation(x1_shuf_patches_3))/2#*abs(x1_shuf_patches_2)
					if self.k_lut > 5:
						x0s4_pos=(1+BiRealNetBinaryActivation(x0_shuf_patches_4))/2#*abs(x0_shuf_patches_2)
						x0s4_neg=(1-BiRealNetBinaryActivation(x0_shuf_patches_4))/2#*abs(x0_shuf_patches_2)
						x1s4_pos=(1+BiRealNetBinaryActivation(x1_shuf_patches_4))/2#*abs(x1_shuf_patches_2)
						x1s4_neg=(1-BiRealNetBinaryActivation(x1_shuf_patches_4))/2#*abs(x1_shuf_patches_2)
		

					if self.k_lut == 1:
						self.out=         K.dot(x0_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)

					if self.k_lut == 2:
						self.out=         K.dot(x0_pos*x0s0_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos, tf.reshape(self.clamped_c_param[4], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg, tf.reshape(self.clamped_c_param[5], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos, tf.reshape(self.clamped_c_param[6], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg, tf.reshape(self.clamped_c_param[7], [-1, self.nfilters])*self.pruning_mask)

					if self.k_lut == 3:
						self.out=         K.dot(x0_pos*x0s0_pos*x0s1_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos, tf.reshape(self.clamped_c_param[4], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg, tf.reshape(self.clamped_c_param[5], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos, tf.reshape(self.clamped_c_param[6], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg, tf.reshape(self.clamped_c_param[7], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos, tf.reshape(self.clamped_c_param[8], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg, tf.reshape(self.clamped_c_param[9], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos, tf.reshape(self.clamped_c_param[10], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg, tf.reshape(self.clamped_c_param[11], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos, tf.reshape(self.clamped_c_param[12], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg, tf.reshape(self.clamped_c_param[13], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos, tf.reshape(self.clamped_c_param[14], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg, tf.reshape(self.clamped_c_param[15], [-1, self.nfilters])*self.pruning_mask)

					if self.k_lut == 4:
						self.out=         K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_c_param[4], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_c_param[5], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_c_param[6], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_c_param[7], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_c_param[8], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_c_param[9], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_c_param[10], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_c_param[11], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos, tf.reshape(self.clamped_c_param[12], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg, tf.reshape(self.clamped_c_param[13], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos, tf.reshape(self.clamped_c_param[14], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg, tf.reshape(self.clamped_c_param[15], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_c_param[16], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_c_param[17], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_c_param[18], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_c_param[19], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_c_param[20], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_c_param[21], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_c_param[22], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_c_param[23], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_c_param[24], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_c_param[25], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_c_param[26], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_c_param[27], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos, tf.reshape(self.clamped_c_param[28], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg, tf.reshape(self.clamped_c_param[29], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos, tf.reshape(self.clamped_c_param[30], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg, tf.reshape(self.clamped_c_param[31], [-1, self.nfilters])*self.pruning_mask)

					if self.k_lut == 5:
						self.out=         K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[4], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[5], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[6], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[7], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[8], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[9], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[10], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[11], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[12], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[13], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[14], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[15], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[16], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[17], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[18], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[19], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[20], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[21], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[22], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[23], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[24], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[25], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[26], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[27], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos, tf.reshape(self.clamped_c_param[28], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg, tf.reshape(self.clamped_c_param[29], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos, tf.reshape(self.clamped_c_param[30], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg, tf.reshape(self.clamped_c_param[31], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[32], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[33], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[34], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[35], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[36], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[37], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[38], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[39], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[40], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[41], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[42], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[43], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[44], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[45], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[46], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[47], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[48], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[49], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[50], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[51], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[52], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[53], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[54], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[55], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[56], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[57], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[58], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[59], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos, tf.reshape(self.clamped_c_param[60], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg, tf.reshape(self.clamped_c_param[61], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos, tf.reshape(self.clamped_c_param[62], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg, tf.reshape(self.clamped_c_param[63], [-1, self.nfilters])*self.pruning_mask)

					if self.k_lut == 6:
						self.out=         K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[0], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[1], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[2], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[3], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[4], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[5], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[6], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[7], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[8], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[9], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[10], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[11], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[12], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[13], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[14], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[15], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[16], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[17], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[18], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[19], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[20], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[21], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[22], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[23], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[24], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[25], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[26], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[27], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[28], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[29], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[30], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_pos*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[31], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[32], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[33], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[34], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[35], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[36], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[37], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[38], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[39], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[40], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[41], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[42], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[43], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[44], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[45], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[46], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_pos*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[47], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[48], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[49], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[50], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[51], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[52], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[53], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[54], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_pos*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[55], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[56], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[57], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[58], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_pos*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[59], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_pos, tf.reshape(self.clamped_c_param[60], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_pos*x0s4_neg, tf.reshape(self.clamped_c_param[61], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_pos, tf.reshape(self.clamped_c_param[62], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x0_neg*x0s0_neg*x0s1_neg*x0s2_neg*x0s3_neg*x0s4_neg, tf.reshape(self.clamped_c_param[63], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[64], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[65], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[66], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[67], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[68], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[69], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[70], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[71], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[72], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[73], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[74], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[75], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[76], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[77], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[78], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[79], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[80], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[81], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[82], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[83], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[84], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[85], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[86], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[87], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[88], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[89], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[90], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[91], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[92], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[93], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[94], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_pos*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[95], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[96], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[97], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[98], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[99], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[100], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[101], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[102], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[103], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[104], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[105], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[106], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[107], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[108], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[109], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[110], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_pos*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[111], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[112], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[113], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[114], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[115], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[116], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[117], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[118], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_pos*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[119], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[120], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[121], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[122], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_pos*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[123], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_pos, tf.reshape(self.clamped_c_param[124], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_pos*x1s4_neg, tf.reshape(self.clamped_c_param[125], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_pos, tf.reshape(self.clamped_c_param[126], [-1, self.nfilters])*self.pruning_mask)
						self.out=self.out+K.dot(x1_neg*x1s0_neg*x1s1_neg*x1s2_neg*x1s3_neg*x1s4_neg, tf.reshape(self.clamped_c_param[127], [-1, self.nfilters])*self.pruning_mask)

				else:
					x_expanded=0
					for l in range(self.levels):
						x_in=x[l,:,:,:,:]
						x_expanded=x_expanded+x_in
					self.out=K.conv2d(x_expanded, kernel=self.clamped_w*tf.reshape(self.pruning_mask, [self.k, self.k, self.ch_in, self.nfilters]), padding=self.padding,strides=self.strides )
			elif self.levels==3:
				if self.LUT==True:
					x_pos=(1+x)/2
					x_neg=(1-x)/2
					self.out=K.conv2d(x_pos[0,:,:,:,:]*x_pos[1,:,:,:,:]*x_pos[2,:,:,:,:], kernel=self.clamped_w1, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_pos[0,:,:,:,:]*x_pos[1,:,:,:,:]*x_neg[2,:,:,:,:], kernel=self.clamped_w2, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_pos[0,:,:,:,:]*x_neg[1,:,:,:,:]*x_pos[2,:,:,:,:], kernel=self.clamped_w3, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_pos[0,:,:,:,:]*x_neg[1,:,:,:,:]*x_neg[2,:,:,:,:], kernel=self.clamped_w4, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*x_pos[1,:,:,:,:]*x_pos[2,:,:,:,:], kernel=self.clamped_w5, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*x_pos[1,:,:,:,:]*x_neg[2,:,:,:,:], kernel=self.clamped_w6, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*x_neg[1,:,:,:,:]*x_pos[2,:,:,:,:], kernel=self.clamped_w7, padding=self.padding,strides=self.strides )
					self.out=self.out+K.conv2d(x_neg[0,:,:,:,:]*x_neg[1,:,:,:,:]*x_neg[2,:,:,:,:], kernel=self.clamped_w8, padding=self.padding,strides=self.strides )
				else:
					x_expanded=0
					for l in range(self.levels):
						x_in=x[l,:,:,:,:]
						x_expanded=x_expanded+x_in
					self.out=K.conv2d(x_expanded, kernel=self.clamped_w, padding=self.padding,strides=self.strides )


		self.output_dim=self.out.get_shape()
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])


"""
def binarize(x):
    #Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    g = tf.get_default_graph()
    with ops.name_scope("Binarized") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            x=tf.clip_by_value(x,-1,1)
            return tf.sign(x)

class Residual_sign(Layer):
	def __init__(self, levels=1,**kwargs):
		self.levels=levels
		super(Residual_sign, self).__init__(**kwargs)
	def build(self, input_shape):
		ars=np.arange(self.levels)+1.0
		ars=ars[::-1]
		self.means=ars/np.sum(ars)
		self.means=tf.Variable(self.means,dtype=tf.float32)
		K.get_session().run(tf.variables_initializer([self.means]))
		self.trainable_weights=[self.means]

	def call(self, x,mask=None):
		resid = x
		out_bin=0
		for l in range(self.levels):
			out=binarize(resid)*K.abs(self.means[l])
			out_bin=out_bin+out
			resid=resid-out
		return out_bin

	def compute_output_shape(self,input_shape):
		return input_shape
	def set_means(self,X):
		means=np.zeros((self.levels))
		means[0]=1
		resid=np.clip(X,-1,1)
		approx=0
		for l in range(self.levels):
			m=np.mean(np.absolute(resid))
			out=np.sign(resid)*m
			approx=approx+out
			resid=resid-out
			means[l]=m
			err=np.mean((approx-np.clip(X,-1,1))**2)

		means=means/np.sum(means)
		sess=K.get_session()
		sess.run(self.means.assign(means))

class binary_conv(Layer):
	def __init__(self,nfilters,ch_in,k,padding,**kwargs):
		self.nfilters=nfilters
		self.ch_in=ch_in
		self.k=k
		self.padding=padding
		super(binary_conv,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.k*self.k*self.ch_in)
		w = tf.random_normal(shape=[self.k,self.k,self.ch_in,self.nfilters], mean=0.0, stddev=stdv, dtype=tf.float32)
		self.w=K.variable(w)
		self.gamma=K.variable([1.0])
		self.trainable_weights=[self.w,self.gamma]
	def call(self, x,mask=None):
		constraint_gamma=K.abs(self.gamma)
		self.clamped_w=constraint_gamma*binarize(self.w)
		self.out=K.conv2d(x, kernel=self.clamped_w, padding=self.padding)#tf.nn.convolution(x, filter=self.clamped_w , padding=self.padding)
		self.output_dim=self.out.get_shape()
		#self.out=Convolution2D(filters=32, kernel_size=(3,3), strides=(1, 1), padding='valid', use_bias=False)(x)
		return self.out
	def  compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1],self.output_dim[2],self.output_dim[3])

class binary_dense(Layer):
	def __init__(self,n_in,n_out,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		super(binary_dense,self).__init__(**kwargs)
	def build(self, input_shape):
		stdv=1/np.sqrt(self.n_in)
		w = tf.random_normal(shape=[self.n_in,self.n_out], mean=0.0, stddev=stdv, dtype=tf.float32)
		self.w=K.variable(w)
		self.gamma=K.variable([1.0])
		self.trainable_weights=[self.w,self.gamma]
	def call(self, x, mask=None):
		constraint_gamma=K.abs(self.gamma)
		self.clamped_w=constraint_gamma*binarize(self.w)
		self.out=K.dot(x, self.clamped_w)
		self.output_dim=self.out.get_shape()
		return self.out
	def  compute_output_shape(self,input_shape):
		return (input_shape[0], self.output_dim[1])
"""
class my_flat(Layer):
	def __init__(self,**kwargs):
		super(my_flat,self).__init__(**kwargs)
	def build(self, input_shape):
		return

	def call(self, x, mask=None):
		self.out=tf.reshape(x,[-1,np.prod(x.get_shape().as_list()[1:])])
		return self.out
	def  compute_output_shape(self,input_shape):
		shpe=(input_shape[0],int(np.prod(input_shape[1:])))
		return shpe
