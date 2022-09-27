import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10,mnist
import tensorflow.keras.utils

def get_dataset():

	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	# convert class vectors to binary class matrices
	X_train = X_train.reshape(-1,784)
	X_test = X_test.reshape(-1,784)
	
	X_train=X_train.astype(np.float32)
	X_test=X_test.astype(np.float32)
	Y_train = np_utils.to_categorical(y_train, 10)
	Y_test = np_utils.to_categorical(y_test, 10)
	X_train /= 256
	X_test /= 256
	X_train=2*X_train-1
	X_test=2*X_test-1

	return (X_train, Y_train, y_train, X_test, Y_test, y_test)