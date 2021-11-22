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

def binarize(x):
    '''Element-wise rounding to the closest integer with full gradient propagation.
    A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    '''
    clipped = K.clip(x,-1,1)
    rounded = K.sign(clipped)
    return clipped + K.stop_gradient(rounded - clipped)

def genkernel_tf (K,i):
    
    #Kernel Generating function 
    eye1=np.identity(2**i).astype(np.float32)
    eye2=np.identity(2**(K-1-i)).astype(np.float32)
    
    mat=np.kron(np.ones([2,2]).astype(np.float32),eye2)
    
    mat=0.5*np.kron(eye1,mat)
    
    return mat

def genkernel_mat_tf (K,mask,tile_size):
    
    out=np.kron(np.ones([tile_size[0],tile_size[1],1,1]).astype(np.float32),np.identity(2**K).astype(np.float32))
    
    for i in range(K):
        #Mask Reshaping and tiling 
        sel_mask=tf.reshape(mask[i],(1,1,tile_size[0],tile_size[1]))
        sel_mask=tf.transpose(sel_mask,(2,3,0,1))
        sel_mask=tf.tile(sel_mask,(1,1,2**K,2**K))
        
        #Matrix of "Kernels" with shape=tile.size
        a=tf.reshape(genkernel(K=K,i=i),(1,1,2**K,2**K))
        a=tf.tile(a,(tile_size[0],tile_size[1],1,1))
        
        #Matrix of "Identity Matrixes" with shape=tile.size
        b=tf.reshape(np.identity(2**K).astype(np.float32),(1,1,2**K,2**K))
        b=tf.tile(b,(tile_size[0],tile_size[1],1,1))
        
        c=tf.where(sel_mask,a,b)
        out=tf.matmul(out,c)
        
    return out

def dummy_kernel_mat (K,num_ops):
    out=np.kron(np.ones([num_ops,1,1]).astype(np.float32),np.identity(2**K).astype(np.float32))
    
    return out

class Residual_sign(Layer):
    def __init__(self, levels=1,trainable=True,**kwargs):
        self.levels=levels
        self.trainable=trainable
        super(Residual_sign, self).__init__(**kwargs)
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
                out=binarize(resid)*abs(self.means[l])
                #out_bin=out_bin+out
                out_bin=out_bin+out#no gamma per level
                resid=resid-out
        elif self.levels==2:
            out=binarize(resid)*abs(self.means[0])
            out_bin=out
            resid=resid-out
            out=binarize(resid)*abs(self.means[1])
            out_bin=tf.stack([out_bin,out])
            resid=resid-out
        elif self.levels==3:
            out=binarize(resid)*abs(self.means[0])
            out_bin1=out
            resid=resid-out
            out=binarize(resid)*abs(self.means[1])
            out_bin2=out
            resid=resid-out
            out=binarize(resid)*abs(self.means[2])
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

class binary_conv(Layer):
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
		super(binary_conv,self).__init__(**kwargs)
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
		#self.gamma=K.variable(1.0)
		self.gamma = self.add_weight(
			name='Variable',
			initializer=tf.keras.initializers.Constant(value=1.0),
			trainable=True
		)

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
			if self.LUT==True:
				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)
				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.gamma]
			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.k,self.k,self.ch_in,self.nfilters]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]

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

		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)

		if self.levels==1 or self.first_layer==True:
			if self.BINARY==False:
				self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
			else:
				self.clamped_w=constraint_gamma*binarize(self.w)
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

					self.clamped_c_param =constraint_gamma*K.clip(self.c_param,  -1,1)

				elif self.LOGIC_SHRINKAGE==True:

					self.clamped_c_param= constraint_gamma*binarize(tf.concat([c_mat1, c_mat2], axis=0))

				else:

					self.clamped_c_param =constraint_gamma*binarize(self.c_param)

			else:
				if self.BINARY==False:
					self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
				else:
					self.clamped_w=constraint_gamma*binarize(self.w)
		elif self.levels==3:
			if self.LUT==True:
				self.clamped_w1=constraint_gamma*binarize(self.w1)
				self.clamped_w2=constraint_gamma*binarize(self.w2)
				self.clamped_w3=constraint_gamma*binarize(self.w3)
				self.clamped_w4=constraint_gamma*binarize(self.w4)
				self.clamped_w5=constraint_gamma*binarize(self.w5)
				self.clamped_w6=constraint_gamma*binarize(self.w6)
				self.clamped_w7=constraint_gamma*binarize(self.w7)
				self.clamped_w8=constraint_gamma*binarize(self.w8)

			else:
				self.clamped_w=constraint_gamma*binarize(self.w)

		if keras.__version__[0]=='2':

			if self.levels==1 or self.first_layer==True:
				self.out=K.conv2d(x, kernel=self.clamped_w*tf.reshape(self.pruning_mask, [self.k, self.k, self.ch_in, self.nfilters]), padding=self.padding,strides=self.strides )
			elif self.levels==2:
				if self.LUT==True:
					x0_patches = tf.extract_image_patches(x[0,:,:,:,:],
						[1, self.k, self.k, 1],
						[1, self.strides[0], self.strides[1], 1], [1, 1, 1, 1],
						padding=self.PADDING)
					x1_patches = tf.extract_image_patches(x[1,:,:,:,:],
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
	
					x0_pos=(1+binarize(x0_patches))/2*abs(x0_patches)
					x0_neg=(1-binarize(x0_patches))/2*abs(x0_patches)
					x1_pos=(1+binarize(x1_patches))/2*abs(x1_patches)
					x1_neg=(1-binarize(x1_patches))/2*abs(x1_patches)
					if self.k_lut > 1:
						x0s0_pos=(1+binarize(x0_shuf_patches_0))/2#*abs(x0_shuf_patches_0)
						x0s0_neg=(1-binarize(x0_shuf_patches_0))/2#*abs(x0_shuf_patches_0)
						x1s0_pos=(1+binarize(x1_shuf_patches_0))/2#*abs(x1_shuf_patches_0)
						x1s0_neg=(1-binarize(x1_shuf_patches_0))/2#*abs(x1_shuf_patches_0)
					if self.k_lut > 2:
						x0s1_pos=(1+binarize(x0_shuf_patches_1))/2#*abs(x0_shuf_patches_1)
						x0s1_neg=(1-binarize(x0_shuf_patches_1))/2#*abs(x0_shuf_patches_1)
						x1s1_pos=(1+binarize(x1_shuf_patches_1))/2#*abs(x1_shuf_patches_1)
						x1s1_neg=(1-binarize(x1_shuf_patches_1))/2#*abs(x1_shuf_patches_1)
					if self.k_lut > 3:
						x0s2_pos=(1+binarize(x0_shuf_patches_2))/2#*abs(x0_shuf_patches_2)
						x0s2_neg=(1-binarize(x0_shuf_patches_2))/2#*abs(x0_shuf_patches_2)
						x1s2_pos=(1+binarize(x1_shuf_patches_2))/2#*abs(x1_shuf_patches_2)
						x1s2_neg=(1-binarize(x1_shuf_patches_2))/2#*abs(x1_shuf_patches_2)
					if self.k_lut > 4:
						x0s3_pos=(1+binarize(x0_shuf_patches_3))/2#*abs(x0_shuf_patches_2)
						x0s3_neg=(1-binarize(x0_shuf_patches_3))/2#*abs(x0_shuf_patches_2)
						x1s3_pos=(1+binarize(x1_shuf_patches_3))/2#*abs(x1_shuf_patches_2)
						x1s3_neg=(1-binarize(x1_shuf_patches_3))/2#*abs(x1_shuf_patches_2)
					if self.k_lut > 5:
						x0s4_pos=(1+binarize(x0_shuf_patches_4))/2#*abs(x0_shuf_patches_2)
						x0s4_neg=(1-binarize(x0_shuf_patches_4))/2#*abs(x0_shuf_patches_2)
						x1s4_pos=(1+binarize(x1_shuf_patches_4))/2#*abs(x1_shuf_patches_2)
						x1s4_neg=(1-binarize(x1_shuf_patches_4))/2#*abs(x1_shuf_patches_2)
		

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

class binary_dense(Layer):
	def __init__(self,n_in,n_out,levels=1,k_lut=4,first_layer=False,LUT=True,REG=None,BINARY=True,LOGIC_SHRINKAGE=False,custom_rand_seed=0,**kwargs):
		self.n_in=n_in
		self.n_out=n_out
		self.levels=levels
		self.k_lut=k_lut
		self.LUT=LUT
		self.REG=REG
		self.BINARY=BINARY
		self.LOGIC_SHRINKAGE=LOGIC_SHRINKAGE
		#self.custom_rand_seed=custom_rand_seed
		self.first_layer=first_layer
		super(binary_dense,self).__init__(**kwargs)
	def get_config(self):
		config = super().get_config().copy()
		config.update({
			'n_in': self.n_in,
			'n_out': self.n_out,
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
				shape=(self.n_in, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 2:
			self.rand_map_1 = self.add_weight(name='rand_map_1', 
				shape=(self.n_in, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 3:
			self.rand_map_2 = self.add_weight(name='rand_map_2', 
				shape=(self.n_in, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 4:
			self.rand_map_3 = self.add_weight(name='rand_map_3', 
				shape=(self.n_in, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
				trainable=False) # Randomisation map for subsequent input connections
		if self.k_lut > 5:
			self.rand_map_4 = self.add_weight(name='rand_map_4', 
				shape=(self.n_in, 1),
				initializer=keras.initializers.Constant(value=np.random.randint(self.n_in, size=[self.n_in, 1])),
				trainable=False) # Randomisation map for subsequent input connections

		stdv=1/np.sqrt(self.n_in)
		#self.gamma=K.variable(1.0)
		self.gamma = self.add_weight(
			name='Variable',
			initializer=tf.keras.initializers.Constant(value=1.0),
			trainable=True
		)

		if self.levels==1 or self.first_layer==True:
			#w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
			#self.w=K.variable(w)
			#self._trainable_weights=[self.w,self.gamma]

			if self.REG==True:
				self.w = self.add_weight(
					name='Variable_1',
					shape=[self.n_in,self.n_out],
					initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
					regularizer=tf.keras.regularizers.l2(5e-7),
					trainable=True
				)
			else:
				self.w = self.add_weight(
					name='Variable_1',
					shape=[self.n_in,self.n_out],
					initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
					trainable=True
				)

		elif self.levels==2:
			if self.LUT==True:
				#c_param = np.random.normal(loc=0.0, scale=stdv,size=[(2**self.k_lut)*self.levels,self.n_in,self.n_out]).astype(np.float32)
				#self.c_param =K.variable(c_param)
				#self._trainable_weights=[self.c_param, self.gamma]

				if self.REG==True:
					self.c_param = self.add_weight(
						name='Variable_1',
						shape=[(2**self.k_lut)*self.levels,self.n_in,self.n_out],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						regularizer=tf.keras.regularizers.l2(5e-7),
						trainable=True
					)
				else:
					self.c_param = self.add_weight(
						name='Variable_1',
						shape=[(2**self.k_lut)*self.levels,self.n_in,self.n_out],
						initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=stdv, seed=None),
						trainable=True
					)

			else:
				#w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				#self.w=K.variable(w)
				#self._trainable_weights=[self.w,self.gamma]

				if self.REG==True:
					self.w = self.add_weight(name='Variable_1',
						shape=[self.n_in,self.n_out],
						initializer=tf.keras.initializers.RandomNormal(
							mean=0.0, stddev=stdv, seed=None
						),
						regularizer=tf.keras.regularizers.l2(5e-7),
						trainable=True
					)
				else:
					self.w = self.add_weight(name='Variable_1',
						shape=[self.n_in,self.n_out],
						initializer=tf.keras.initializers.RandomNormal(
							mean=0.0, stddev=stdv, seed=None
						),
						trainable=True
					)

		elif self.levels==3:
			if self.LUT==True:
				w1 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w2 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w3 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w4 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w5 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w6 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w7 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				w8 = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				self.w1=K.variable(w1)
				self.w2=K.variable(w2)
				self.w3=K.variable(w3)
				self.w4=K.variable(w4)
				self.w5=K.variable(w5)
				self.w6=K.variable(w6)
				self.w7=K.variable(w7)
				self.w8=K.variable(w8)

				self.trainable_weights=[self.w1,self.w2,self.w3,self.w4,self.w5,self.w6,self.w7,self.w8,self.gamma]
			else:
				w = np.random.normal(loc=0.0, scale=stdv,size=[self.n_in,self.n_out]).astype(np.float32)
				self.w=K.variable(w)
				self.trainable_weights=[self.w,self.gamma]

		if self.LUT == True: # Construct shrinkage maps

			value=dummy_kernel_mat(K=self.k_lut,num_ops=self.n_in*self.n_out)
			self.shrinkage_map = self.add_weight(name='shrinkage_map',
				shape=value.shape,
				initializer=keras.initializers.Constant(value),
				trainable=False)

		self.pruning_mask = self.add_weight(name='pruning_mask',
			shape=(self.n_in,self.n_out),
			initializer=keras.initializers.Constant(value=np.ones((self.n_in,self.n_out))),
			trainable=False) # LUT pruning based on whether inputs get repeated

	def call(self, x,mask=None):

		#np.random.seed(self.custom_rand_seed)
		#tf.set_random_seed(self.rand_seed)
		#random.seed(self.rand_seed)

		constraint_gamma=K.abs(self.gamma)#K.clip(self.gamma,0.01,10)
		if self.levels==1 or self.first_layer==True:
			if self.BINARY==False:
				self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
			else:
				self.clamped_w=constraint_gamma*binarize(self.w)
			self.out=K.dot(x,self.clamped_w)
		elif self.levels==2:
			if self.LUT==True:

				if self.LOGIC_SHRINKAGE==True:

					#SHRINKAGE
                #Stacking c coef	ficients to create c_mat1 and c_mat2 tensors
					c_mat1=self.c_param[0 : 2**self.k_lut, :, :]
					c_mat2=self.c_param[2**self.k_lut : (2**self.k_lut)*2, :, :]

					c_mat1=tf.reshape(c_mat1,(1,2**self.k_lut,self.n_in*self.n_out))                
					c_mat2=tf.reshape(c_mat2,(1,2**self.k_lut,self.n_in*self.n_out))
       
					c_mat1=tf.transpose(c_mat1,(2,0,1))
					c_mat2=tf.transpose(c_mat2,(2,0,1))
            
					c_mat1=tf.matmul(c_mat1,self.shrinkage_map)
					c_mat2=tf.matmul(c_mat2,self.shrinkage_map)
					
					c_mat1=tf.transpose(c_mat1,(2,0,1))
					c_mat2=tf.transpose(c_mat2,(2,0,1))

					c_mat1 = tf.reshape(c_mat1, (2**self.k_lut, self.n_in, self.n_out))
					c_mat2 = tf.reshape(c_mat2, (2**self.k_lut, self.n_in, self.n_out))
                
				if self.BINARY==False:

					self.clamped_c_param= constraint_gamma*K.clip(self.c_param, -1,1)

				elif self.LOGIC_SHRINKAGE==True:

					self.clamped_c_param= constraint_gamma*binarize(tf.concat([c_mat1, c_mat2], axis=0))

				else:

					self.clamped_c_param= constraint_gamma*binarize(self.c_param)
				
				# Special hack for randomising the subsequent input connections: tensorflow does not support advanced matrix indexing
				shuf_x=tf.transpose(x, perm=[2, 0, 1])
				if self.k_lut > 1:
					shuf_x_0 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_0, tf.int32))
					shuf_x_0=tf.transpose(shuf_x_0, perm=[1, 2, 0], name="extra_input_1_tapped")
				
				if self.k_lut > 2:
					shuf_x_1 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_1, tf.int32))
					shuf_x_1=tf.transpose(shuf_x_1, perm=[1, 2, 0], name="extra_input_2_tapped")

				if self.k_lut > 3:
					shuf_x_2 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_2, tf.int32))
					shuf_x_2=tf.transpose(shuf_x_2, perm=[1, 2, 0], name="extra_input_3_tapped")

				if self.k_lut > 4:
					shuf_x_3 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_3, tf.int32))
					shuf_x_3=tf.transpose(shuf_x_3, perm=[1, 2, 0], name="extra_input_4_tapped")

				if self.k_lut > 5:
					shuf_x_4 = tf.gather_nd(shuf_x, tf.cast(self.rand_map_4, tf.int32))
					shuf_x_4=tf.transpose(shuf_x_4, perm=[1, 2, 0], name="extra_input_5_tapped")
			
				x_pos=(1+binarize(x))/2*abs(x)
				x_neg=(1-binarize(x))/2*abs(x)
				if self.k_lut > 1:
					xs0_pos=(1+binarize(shuf_x_0))/2#*abs(shuf_x_0)
					xs0_neg=(1-binarize(shuf_x_0))/2#*abs(shuf_x_0)
				if self.k_lut > 2:
					xs1_pos=(1+binarize(shuf_x_1))/2#*abs(shuf_x_1)
					xs1_neg=(1-binarize(shuf_x_1))/2#*abs(shuf_x_1)
				if self.k_lut > 3:
					xs2_pos=(1+binarize(shuf_x_2))/2#*abs(shuf_x_2)
					xs2_neg=(1-binarize(shuf_x_2))/2#*abs(shuf_x_2)
				if self.k_lut > 4:
					xs3_pos=(1+binarize(shuf_x_3))/2#*abs(shuf_x_2)
					xs3_neg=(1-binarize(shuf_x_3))/2#*abs(shuf_x_2)
				if self.k_lut > 5:
					xs4_pos=(1+binarize(shuf_x_4))/2#*abs(shuf_x_2)
					xs4_neg=(1-binarize(shuf_x_4))/2#*abs(shuf_x_2)

				if self.k_lut == 1:
					self.out=         K.dot(x_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:],self.clamped_c_param[2]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:],self.clamped_c_param[3]*self.pruning_mask)

				if self.k_lut == 2:
					self.out=         K.dot(x_pos[0,:,:]*xs0_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:],self.clamped_c_param[2]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:],self.clamped_c_param[3]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:],self.clamped_c_param[4]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:],self.clamped_c_param[5]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:],self.clamped_c_param[6]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:],self.clamped_c_param[7]*self.pruning_mask)

				if self.k_lut == 3:
					self.out=         K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:],self.clamped_c_param[2]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:],self.clamped_c_param[3]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:],self.clamped_c_param[4]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:],self.clamped_c_param[5]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:],self.clamped_c_param[6]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:],self.clamped_c_param[7]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:],self.clamped_c_param[8]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:],self.clamped_c_param[9]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:],self.clamped_c_param[10]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:],self.clamped_c_param[11]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:],self.clamped_c_param[12]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:],self.clamped_c_param[13]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:],self.clamped_c_param[14]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:],self.clamped_c_param[15]*self.pruning_mask)

				if self.k_lut == 4:
					self.out=         tf.matmul(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask,name="inter_dot_output_0_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask,name="inter_dot_output_1_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[2]*self.pruning_mask,name="inter_dot_output_2_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[3]*self.pruning_mask,name="inter_dot_output_3_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[4]*self.pruning_mask,name="inter_dot_output_4_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[5]*self.pruning_mask,name="inter_dot_output_5_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[6]*self.pruning_mask,name="inter_dot_output_6_tapped")
					self.out=self.out+tf.matmul(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[7]*self.pruning_mask,name="inter_dot_output_7_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[8]*self.pruning_mask,name="inter_dot_output_8_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[9]*self.pruning_mask,name="inter_dot_output_9_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[10]*self.pruning_mask,name="inter_dot_output_10_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[11]*self.pruning_mask,name="inter_dot_output_11_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[12]*self.pruning_mask,name="inter_dot_output_12_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[13]*self.pruning_mask,name="inter_dot_output_13_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:],self.clamped_c_param[14]*self.pruning_mask,name="inter_dot_output_14_tapped")
					self.out=self.out+tf.matmul(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:],self.clamped_c_param[15]*self.pruning_mask,name="inter_dot_output_15_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[16]*self.pruning_mask,name="inter_dot_output_16_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[17]*self.pruning_mask,name="inter_dot_output_17_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[18]*self.pruning_mask,name="inter_dot_output_18_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[19]*self.pruning_mask,name="inter_dot_output_19_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[20]*self.pruning_mask,name="inter_dot_output_20_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[21]*self.pruning_mask,name="inter_dot_output_21_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[22]*self.pruning_mask,name="inter_dot_output_22_tapped")
					self.out=self.out+tf.matmul(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[23]*self.pruning_mask,name="inter_dot_output_23_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[24]*self.pruning_mask,name="inter_dot_output_24_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[25]*self.pruning_mask,name="inter_dot_output_25_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[26]*self.pruning_mask,name="inter_dot_output_26_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[27]*self.pruning_mask,name="inter_dot_output_27_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[28]*self.pruning_mask,name="inter_dot_output_28_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[29]*self.pruning_mask,name="inter_dot_output_29_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:],self.clamped_c_param[30]*self.pruning_mask,name="inter_dot_output_30_tapped")
					self.out=self.out+tf.matmul(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:],self.clamped_c_param[31]*self.pruning_mask,name="inter_dot_output_31_tapped")

				if self.k_lut == 5:
					self.out=         K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[2]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[3]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[4]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[5]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[6]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[7]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[8]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[9]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[10]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[11]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[12]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[13]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[14]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[15]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[16]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[17]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[18]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[19]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[20]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[21]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[22]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[23]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[24]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[25]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[26]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[27]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[28]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[29]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:],self.clamped_c_param[30]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:],self.clamped_c_param[31]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[32]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[33]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[34]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[35]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[36]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[37]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[38]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[39]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[40]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[41]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[42]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[43]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[44]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[45]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[46]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[47]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[48]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[49]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[50]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[51]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[52]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[53]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[54]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[55]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[56]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[57]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[58]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[59]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[60]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[61]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:],self.clamped_c_param[62]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:],self.clamped_c_param[63]*self.pruning_mask)

				if self.k_lut == 6:
					self.out=         K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[0]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[1]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[2]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[3]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[4]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[5]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[6]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[7]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[8]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[9]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[10]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[11]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[12]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[13]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[14]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[15]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[16]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[17]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[18]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[19]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[20]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[21]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[22]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[23]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[24]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[25]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[26]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[27]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[28]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[29]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[30]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[31]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[32]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[33]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[34]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[35]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[36]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[37]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[38]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[39]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[40]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[41]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[42]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[43]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[44]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[45]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[46]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_pos[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[47]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[48]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[49]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[50]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[51]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[52]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[53]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[54]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_pos[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[55]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[56]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[57]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[58]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_pos[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[59]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[60]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_pos[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[61]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_pos[0,:,:],self.clamped_c_param[62]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[0,:,:]*xs0_neg[0,:,:]*xs1_neg[0,:,:]*xs2_neg[0,:,:]*xs3_neg[0,:,:]*xs4_neg[0,:,:],self.clamped_c_param[63]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[64]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[65]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[66]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[67]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[68]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[69]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[70]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[71]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[72]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[73]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[74]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[75]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[76]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[77]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[78]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[79]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[80]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[81]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[82]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[83]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[84]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[85]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[86]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[87]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[88]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[89]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[90]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[91]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[92]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[93]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[94]*self.pruning_mask)
					self.out=self.out+K.dot(x_pos[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[95]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[96]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[97]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[98]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[99]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[100]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[101]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[102]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[103]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[104]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[105]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[106]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[107]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[108]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[109]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[110]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_pos[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[111]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[112]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[113]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[114]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[115]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[116]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[117]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[118]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_pos[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[119]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[120]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[121]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[122]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_pos[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[123]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[124]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_pos[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[125]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_pos[1,:,:],self.clamped_c_param[126]*self.pruning_mask)
					self.out=self.out+K.dot(x_neg[1,:,:]*xs0_neg[1,:,:]*xs1_neg[1,:,:]*xs2_neg[1,:,:]*xs3_neg[1,:,:]*xs4_neg[1,:,:],self.clamped_c_param[127]*self.pruning_mask)

			else:
				x_expanded=0
				if self.BINARY==False:
					self.clamped_w=constraint_gamma*K.clip(self.w,-1,1)
				else:
					self.clamped_w=constraint_gamma*binarize(self.w)
				for l in range(self.levels):
					x_expanded=x_expanded+x[l,:,:]
				self.out=K.dot(x_expanded,self.clamped_w*self.pruning_mask)
		elif self.levels==3:
			if self.LUT==True:
				self.clamped_w1=constraint_gamma*binarize(self.w1)
				self.clamped_w2=constraint_gamma*binarize(self.w2)
				self.clamped_w3=constraint_gamma*binarize(self.w3)
				self.clamped_w4=constraint_gamma*binarize(self.w4)
				self.clamped_w5=constraint_gamma*binarize(self.w5)
				self.clamped_w6=constraint_gamma*binarize(self.w6)
				self.clamped_w7=constraint_gamma*binarize(self.w7)
				self.clamped_w8=constraint_gamma*binarize(self.w8)
				x_pos=(1+x)/2
				x_neg=(1-x)/2
				self.out=K.dot(x_pos[0,:,:]*x_pos[1,:,:]*x_pos[2,:,:],self.clamped_w1)
				self.out=self.out+K.dot(x_pos[0,:,:]*x_pos[1,:,:]*x_neg[2,:,:],self.clamped_w2)
				self.out=self.out+K.dot(x_pos[0,:,:]*x_neg[1,:,:]*x_pos[2,:,:],self.clamped_w3)
				self.out=self.out+K.dot(x_pos[0,:,:]*x_neg[1,:,:]*x_neg[2,:,:],self.clamped_w4)
				self.out=self.out+K.dot(x_neg[0,:,:]*x_pos[1,:,:]*x_pos[2,:,:],self.clamped_w5)
				self.out=self.out+K.dot(x_neg[0,:,:]*x_pos[1,:,:]*x_neg[2,:,:],self.clamped_w6)
				self.out=self.out+K.dot(x_neg[0,:,:]*x_neg[1,:,:]*x_pos[2,:,:],self.clamped_w7)
				self.out=self.out+K.dot(x_neg[0,:,:]*x_neg[1,:,:]*x_neg[2,:,:],self.clamped_w8)
			else:
				x_expanded=0
				self.clamped_w=constraint_gamma*binarize(self.w)
				for l in range(self.levels):
					x_expanded=x_expanded+x[l,:,:]
				self.out=K.dot(x_expanded,self.clamped_w)
		return self.out
	def  get_output_shape_for(self,input_shape):
		return (input_shape[0], self.n_out)
	def compute_output_shape(self,input_shape):
		return (input_shape[0], self.n_out)



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
