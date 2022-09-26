import os
import numpy as np
import tensorflow as tf
import glob
from logger import logger

#data_dir = "../BinaryCoP_dataset"
#image_count = 133782

data_dir = "../BinaryCoP_balanced_dataset"
# image_count = 138486

train_image_count = 110389
val_image_count = 28097

img_height = 32
img_width = 32

# Following line is to avoid AttributeError: 'Tensor' object has no attribute 'numpy'
tf.enable_eager_execution() 

def get_binarycop_dataset():
    logger.info("Started getting Binarycop dataset.")
    # write input pipeline using tf.data
    
    train_ds = tf.data.Dataset.list_files(str(data_dir+'/train/*/*'), shuffle=False)
    train_ds = train_ds.shuffle(train_image_count, reshuffle_each_iteration=False)
    
    val_ds = tf.data.Dataset.list_files(str(data_dir+'/val/*/*'), shuffle=False)
    val_ds = val_ds.shuffle(val_image_count, reshuffle_each_iteration=False)

    # for debug purpose, print path of 5 images
    #for f in list_ds.take(5):
    #    print(f.numpy())
    
    # the structure of the files can be used to compile a class_names list
    class_names = np.array(sorted([item.split('/')[3] for item in glob.glob(str(data_dir+'/train/*'))]))
    # expected ['CMFD', 'IMFD_Chin', 'IMFD_Mouth_Chin', 'IMFD_Nose_Mouth']
    logger.info("BINARY COP classes are "+str(class_names))

    # # split the dataset into training and validation datasets
    # val_size = int(image_count * 0.2)
    # train_ds = list_ds.skip(val_size)
    # val_ds = list_ds.take(val_size)

    def get_label(file_path):
        # Convert the path to a list of path components
        parts = tf.strings.split(file_path, os.path.sep, result_type='RaggedTensor')
        # The second to last is the class-directory

        bool_values = tf.equal(parts[-2], class_names)
        indices = tf.where(bool_values)
        #one_hot = tf.one_hot(indices, depth=len(class_names))

        # Integer encode the label
        # use indices[0]    for label = [class_number]  (CIFAR10 like)
        # use indices[0][0] for label = class_number    (MNIST like)
        return indices[0]

    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(file_path):
        label = get_label(file_path)
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # print the length of each dataset
    print(tf.data.experimental.cardinality(train_ds).numpy(), "samples in the train dataset")
    print(tf.data.experimental.cardinality(val_ds).numpy(), "samples in the val dataset")

    # Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # for image, label in train_ds.take(10):
    #    print("Image shape: ", image.numpy().shape)
    #    print("Label: ", label.numpy())

    images_list, labels_list = list(zip(*train_ds))
    X_train = np.array([X.numpy() for X in images_list])
    y_train = np.array([y.numpy() for y in labels_list])

    images_list, labels_list = list(zip(*val_ds))
    X_test = np.array([X.numpy() for X in images_list])
    y_test = np.array([y.numpy() for y in labels_list])

    logger.info("Finished getting Binarycop dataset.")

    return (X_train, y_train), (X_test, y_test)

def get_image(image_path, class_name):
    # expected ['correct_mask' 'uncovered_chin' 'uncovered_mouth_nose' 'uncovered_nose']

    logger.info("Started generating test image.")
    # write input pipeline using tf.data
    val_ds = tf.data.Dataset.list_files(image_path, shuffle=False)
    
    # for debug purpose, print path of 1 image
    for f in val_ds.take(1):
        print(f.numpy())

    def decode_img(img):
        # Convert the compressed string to a 3D uint8 tensor
        img = tf.io.decode_jpeg(img, channels=3)
        # Resize the image to the desired size
        return tf.image.resize(img, [img_height, img_width])

    def process_path(file_path):
        label = 0
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img, label

    # Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.
    val_ds = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for image, label in val_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())

    images_list, labels_list = list(zip(*val_ds))
    X_test = np.array([X.numpy() for X in images_list])
    y_test = np.array([y.numpy() for y in labels_list])

    #print(X_test)
    #print(y_test)

    logger.info("Finished generating test image.")

    return (X_test, y_test)

def write_image_bin_file(X, y, out_filename):
    print("write test images into bin files")
    
    X_tr = np.transpose(X, (0,3,1,2))

    fp0 = open(out_filename+"/test_0.bin", "wb") 
    fp0.close()
    fp1 = open(out_filename+"/test_1.bin", "wb") 
    fp1.close()
    fp2 = open(out_filename+"/test_2.bin", "wb") 
    fp2.close()
    fp3 = open(out_filename+"/test_3.bin", "wb") 
    fp3.close()

    for test_idx in range(X_tr.shape[0]):

        r = X_tr[test_idx][0][:][:].flatten().astype(np.uint8)
        g = X_tr[test_idx][1][:][:].flatten().astype(np.uint8)
        b = X_tr[test_idx][2][:][:].flatten().astype(np.uint8)

        label = np.identity(1, dtype=np.uint8)

        if y[test_idx][0] == 0:
            fp0 = open(out_filename+"/test_0.bin", "ab") 
            fp0.write(label.tobytes())
            fp0.write(r.tobytes())
            fp0.write(g.tobytes())
            fp0.write(b.tobytes())
            fp0.close()

        elif y[test_idx][0] == 1:
            fp1 = open(out_filename+"/test_1.bin", "ab") 
            fp1.write(label.tobytes())
            fp1.write(r.tobytes())
            fp1.write(g.tobytes())
            fp1.write(b.tobytes())
            fp1.close()

        elif y[test_idx][0] == 2:
            fp2 = open(out_filename+"/test_2.bin", "ab") 
            fp2.write(label.tobytes())
            fp2.write(r.tobytes())
            fp2.write(g.tobytes())
            fp2.write(b.tobytes())
            fp2.close()

        elif y[test_idx][0] == 3:
            fp3 = open(out_filename+"/test_3.bin", "ab") 
            fp3.write(label.tobytes())
            fp3.write(r.tobytes())
            fp3.write(g.tobytes())
            fp3.write(b.tobytes())
            fp3.close()