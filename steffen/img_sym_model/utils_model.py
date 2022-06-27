import os
import numpy as np
import cv2
import tensorflow as tf
import time
from tensorflow.keras.applications.vgg16 import preprocess_input


def get_data(DEFAULT_PATH, split):

    print('Loading data:')

    train_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_sym_data', 'split' + str(split), 'train')
    train_data = [[], []]
    train_labels = []
    for score_nr in range(4):
        score_path = os.path.join(train_path, 'score' + str(score_nr))
        # os.fsencode(score_path)
        for npz_file in os.listdir(score_path):
            npz_file_path = os.path.join(score_path, npz_file)
            data = np.load(npz_file_path, allow_pickle=True)
            img = cv2.resize(data['image'], (224, 224))
            train_data[0].append(img)
            train_data[1].append(data['symptoms'])
            train_labels.append(score_nr)

    val_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_sym_data', 'split' + str(split), 'validation')
    val_data = [[], []]
    val_labels = []
    for score_nr in range(4):
        score_path = os.path.join(val_path, 'score' + str(score_nr))
        # os.fsencode(score_path)
        for npz_file in os.listdir(score_path):
            npz_file_path = os.path.join(score_path, npz_file)
            data = np.load(npz_file_path, allow_pickle=True)
            img = cv2.resize(data['image'], (224, 224))        
            val_data[0].append(img)
            val_data[1].append(data['symptoms'])
            val_labels.append(score_nr)

    # train_data = np.array(train_data, dtype=object)
    # train_labels = np.array(train_labels, dtype=object)
    # val_data = np.array(val_data, dtype=object)
    # val_labels = np.array(val_labels, dtype=object)

    print('Done')

    return train_data, train_labels, val_data, val_labels


def create_tf_dataset(train_data, train_labels, val_data, val_labels, batch_size):

    print('\nCreating Tensorflow Dataset:')

    train_ds = tf.data.Dataset.from_tensor_slices((
        {'images': train_data[0], 'symptoms': train_data[1]}, 
        train_labels))
    val_ds = tf.data.Dataset.from_tensor_slices((
        {'images': val_data[0], 'symptoms': val_data[1]}, 
        val_labels))

    print('Done')
    print('Applying preprocessing:')
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.map(lambda data, label: (
        # tf.expand_dims(image/255, -1), 
        {'images': preprocess_input(data['images']),
        'symptoms': tf.expand_dims(data['symptoms'], -1)},                   
        tf.one_hot(tf.cast(label, tf.int32), 4)
        )) 
    val_ds = val_ds.map(lambda data, label: (
        # tf.expand_dims(image/255, -1), 
        {'images': preprocess_input(data['images']),  
        'symptoms': tf.expand_dims(data['symptoms'], -1)}, 
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

    train_ds = train_ds.map(lambda data, label: (
            {'images': data_augmentation(data['images'], training=True),
            'symptoms': data['symptoms']},   
            label),
        num_parallel_calls=AUTOTUNE
        )
    val_ds = val_ds.map(lambda data, label: (
            {'images': data_augmentation(data['images'], training=True),  
            'symptoms': data['symptoms']}, 
            label),
        num_parallel_calls=AUTOTUNE
        )

    print('Done')

    return train_ds, val_ds


def fix_layers(model, num_flex_layers: int = 1):
    """
    Receives a model and freezes all layers but the last num_flex_layers ones.

    Arguments:
        model {tensorflow.python.keras.engine.training.Model} -- model

    Keyword Arguments:
        num_flex_layers {int} -- [Number of trainable layers] (default: {1})

    Returns:
        Model -- updated model
    """
    num_layers = len(model.layers)
    for ind, layer in enumerate(model.layers):
        if ind < num_layers - num_flex_layers:
            layer.trainable = False

    return model