import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer

#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(PreprocessingLayer):
  def __init__(self, alphastd, eigval=imagenet_pca['eigval'], eigvec=imagenet_pca['eigvec'], **kwargs):
    self.alphastd = alphastd
    assert eigval.shape == (3,)
    assert eigvec.shape == (3, 3)
    self.eigval = eigval
    self.eigvec = eigvec
    super(Lighting, self).__init__(**kwargs)

  def call(self, inputs):
    if self.alphastd == 0.:
      return inputs
    rnd = np.random.randn(3) * self.alphastd
    rnd = rnd.astype('float32')
    v = rnd
    #old_dtype = np.asarray(inputs).dtype
    v = v * self.eigval
    v = v.reshape((3, 1))
    inc = np.dot(self.eigvec, v).reshape((3,))
    inputs = tf.add(inputs, inc)
    #if old_dtype == np.uint8:
    #  inputs = np.clip(inputs, 0, 255)
    inputs = tf.clip_by_value(inputs, 0, 255)
    #inputs = Image.fromarray(inputs.astype(old_dtype), 'RGB')
    return inputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    return tensor_shape.TensorShape(
      [input_shape[0], self.target_height, self.target_width, input_shape[3]])

  def get_config(self):
    config = {
      'alphastd': self.alphastd,
      'eigval': self.eigval,
      'eigvec': self.eigvec,
    }
    base_config = super(Lighting, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

