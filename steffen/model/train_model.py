import os 
import numpy as np
import tensorflow as tf
import time
from utils import get_data, create_tf_dataset, train_step, test
from model import get_model


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
    train_images, train_labels, val_images, val_labels = get_data(DEFAULT_PATH, split = 0)

    train_ds, val_ds = create_tf_dataset(train_images, train_labels, val_images, val_labels, batch_size = 1)

    # Creating model and aggregators
    model = get_model(trainable_layers=1)

    train_loss_aggregator = []

    test_loss_aggregator = []
    test_accuracy_aggregator = []

    # Conducting first tests before training
    print('Starting tests before training')
    start = time.time()

    test_loss, test_accuracy = test(model, val_ds, loss_function)
    train_loss_aggregator.append(test_loss)
    test_accuracy_aggregator.append(test_accuracy)

    train_loss, train_accuracy = test(model, train_ds, loss_function)
    train_loss_aggregator.append(train_loss)

    end = time.time()

    print(f'Tests took: {round(end-start, 2)}. Accuracy before training: {test_accuracy}')

    print('\nStart training.')
    for epoch in (range(num_epochs)): # tqdm to track progress
        start = time.time()

        epoch_loss_agg = []
            
        #Go through training data and train the model
        for input, target in train_ds:

            # Perform on train step on one batch
            train_loss = train_step(input, target, model, loss_function, optimizer)
            epoch_loss_agg.append(train_loss)

        train_loss_aggregator.append(np.mean(epoch_loss_agg))

        # Test the model on test dataset after training on train dataset
        test_loss, test_accuracy = test(model, val_ds, loss_function)
        test_loss_aggregator.append(test_loss)
        test_accuracy_aggregator.append(test_accuracy)

        print(f'Accuracy: {test_accuracy}')
        end = time.time()
        print(f'Epoch: {epoch+1} took {round(end-start, 2)} seconds.\n')

    print(f'Training complete after {num_epochs} Epochs. Accuracy is {test_accuracy_aggregator[-1]}')

    