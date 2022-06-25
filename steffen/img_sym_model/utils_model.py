import os
import numpy as np
import cv2
import tensorflow as tf
import time
from tensorflow.keras.applications.vgg16 import preprocess_input


def get_data(DEFAULT_PATH, split):

    print('Loading data:')

    train_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_sym_data', 'split' + str(split), 'train')
    train_data = []
    train_labels = []
    for score_nr in range(4):
        score_path = os.path.join(train_path, 'score' + str(score_nr))
        # os.fsencode(score_path)
        for npz_file in os.listdir(score_path):
            npz_file_path = os.path.join(score_path, npz_file)
            data = np.load(npz_file_path)
            img = cv2.resize(data['image'], (224, 224))
            train_data.append([img, data['symptoms']])
            train_labels.append(score_nr)

    val_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_data', 'split' + str(split), 'validation')
    val_images = []
    val_labels = []
    for score_nr in range(4):
        score_path = os.path.join(val_path, 'score' + str(score_nr))
        # os.fsencode(score_path)
        for image_name in os.listdir(score_path):
            img_path = os.path.join(score_path, image_name)
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (224, 224))        
            val_images.append(img)
            val_labels.append(score_nr)

    # train_images = np.array(train_images, dtype=object)
    # train_labels = np.array(train_labels, dtype=object)
    # val_images = np.array(val_images, dtype=object)
    # val_labels = np.array(val_labels, dtype=object)

    print('Done')

    return train_data, train_labels, val_images, val_labels


def create_tf_dataset(train_images, train_labels, val_images, val_labels, batch_size):

    print('\nCreating Tensorflow Dataset:')

    train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    print('Done')
    print('Applying preprocessing:')
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda image, label: (
        # tf.expand_dims(image/255, -1), 
        preprocess_input(image),                   
        tf.one_hot(tf.cast(label, tf.int32), 4)
        )) 
    val_ds = val_ds.map(lambda image, label: (
        # tf.expand_dims(image/255, -1), 
        preprocess_input(image),  
        tf.one_hot(tf.cast(label, tf.int32), 4)
        ))

    
    train_ds = train_ds.shuffle(
                            train_ds.cardinality(), 
                            reshuffle_each_iteration=True
                        ).batch(batch_size).prefetch(AUTOTUNE)

    val_ds = val_ds.shuffle(
                            train_ds.cardinality(), 
                            reshuffle_each_iteration=True
                        ).batch(batch_size).prefetch(AUTOTUNE)


    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal_and_vertical'),
        tf.keras.layers.RandomRotation(0.2),
        # tf.keras.layers.RandomBrightness(0.2, value_range=[0.0, 1.0]),
        tf.keras.layers.RandomContrast(0.2),
    ])

    train_ds = train_ds.map(lambda image, label: (
        data_augmentation(image, training=True),  label),
        num_parallel_calls=AUTOTUNE
        )
    val_ds = val_ds.map(lambda image, label: (
        data_augmentation(image, training=True),  label),
        num_parallel_calls=AUTOTUNE
        )

    print('Done')

    return train_ds, val_ds