import os 
import numpy as np
import tensorflow as tf
import time
from utils_model import get_data, create_tf_dataset#, train_step, test, save_loss_and_accuracy
from model import MultimodalModel
from tensorflow.keras.callbacks import EarlyStopping


if __name__ == '__main__':

    os.chdir('..')
    os.chdir('..')
    DEFAULT_PATH = os.getcwd()
    num_epochs = 10
    learning_rate = 0.01

    tf.keras.backend.clear_session()

    # Create loss function and optimizer
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    # Get the dataset
    train_data, train_labels, val_data, val_labels = get_data(DEFAULT_PATH, split = 0)

    train_ds, val_ds = create_tf_dataset(train_data, train_labels, val_data, val_labels, batch_size = 16)

    # Creating model and aggregators
    model = MultimodalModel()

    model.compile(
        optimizer='adam',
        loss = 'categorical_crossentropy',
        metrics=['accuracy']
    )

    es = EarlyStopping(
        monitor='val_loss', 
        mode='min', 
        patience=10,  
        verbose=1,
        restore_best_weights=True
        )

    model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=[es])