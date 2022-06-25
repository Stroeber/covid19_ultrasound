import os
import numpy as np
import cv2
import tensorflow as tf
import time
from tensorflow.keras.applications.vgg16 import preprocess_input


def get_data(DEFAULT_PATH, split):

    print('Loading data:')

    train_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_data', 'split' + str(split), 'train')
    train_images = []
    train_labels = []
    for score_nr in range(4):
        score_path = os.path.join(train_path, 'score' + str(score_nr))
        # os.fsencode(score_path)
        for image_name in os.listdir(score_path):
            img_path = os.path.join(score_path, image_name)
            img = cv2.imread(img_path, 1)
            img = cv2.resize(img, (224, 224))
            train_images.append(img)
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

    return train_images, train_labels, val_images, val_labels


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


def train_step(input, label, model, loss_func, optimizer):
    '''
    Trains the model on one batch of data.
    
    Args:
        input: One batch of image data
        label: One batch ob corresponding labels
        model: The Model to be trained
        loss_func: The loss function to compute the loss
        optimizer: The optimizer to apply the gradients
    '''
    with tf.GradientTape() as tape:
        prediction = model(input, training=True) # Get the models prediction
        loss = loss_func(label, prediction) # Compute the loss by comparing prediction to correct label
    gradients = tape.gradient(loss, model.trainable_variables) # Compute the gradients using the loss
    optimizer.apply_gradients(zip(gradients, model.trainable_variables)) # Apply the gradients using the optimizer
    return loss


def test(model, test_data, loss_function):
    '''
    Computes the accuracy and loss of the given data on the given model

    Args:
        model: The Tensorflow model to be tested
        test_data: The data the model will be tested (includes image and label)
        loss_function: Loss function to compute the loss
    '''

    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        prediction = model(input) # Get the models prediction
        
        sample_test_loss = loss_function(target, prediction) # Compute the loss
        test_loss_aggregator.append(sample_test_loss.numpy())

        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1) # Compute the accuracy of one batch
        test_accuracy_aggregator.append(np.mean(sample_test_accuracy)) # Compute the mean of one batch

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

    return test_loss, test_accuracy


def save_loss_and_accuracy(train_loss, test_loss, accuracy, model_name, split_nr, DEFAULT_PATH, timestr):
    os.chdir(os.path.join(DEFAULT_PATH, 'steffen', 'model'))
    model_data_path = os.path.join(os.getcwd(), 'model_data')
    if not os.path.exists(model_data_path):
        os.mkdir(model_data_path)
    os.chdir(model_data_path)
    # timestr = time.strftime("%Y%m%d-%H%M%S")
    savefile_name = f'{model_name}_split{split_nr}_{timestr}' #Sample: 'VGG16_split4_20220623-143345'
    np.savez(savefile_name, train_loss = train_loss, test_loss = test_loss, accuracy=accuracy)