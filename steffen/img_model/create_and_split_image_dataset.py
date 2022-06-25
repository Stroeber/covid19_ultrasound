import os
import pandas as pd
import numpy as np
from utils_data import delete_img_data_folder, create_images_from_videos, cross_val_split, store_split


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    DEFAULT_PATH = os.getcwd()
    os.chdir('data')

    metadata_raw = pd.read_csv("dataset_metadata.csv")

    metadata = metadata_raw.filter(items=['Current location','Filename', 'Lung Severity Score'])
    metadata = metadata.replace('n/A', np.nan)
    metadata = metadata.dropna(subset=['Lung Severity Score'])

    grouped_images, grouped_scores = create_images_from_videos(DEFAULT_PATH, metadata)

    indices =  np.arange(len(grouped_images))
    np.random.shuffle(indices)
    split_val_indices = np.array_split(indices, 5)

    delete_img_data_folder(DEFAULT_PATH)

    print('\nStart storing splits:')
    for split_nr in range(len(split_val_indices)):

        images_train, images_val, labels_train, labels_val = cross_val_split(
            indices, 
            split_val_indices[split_nr], 
            grouped_images, 
            grouped_scores, 
            split_nr
            )

        store_split(
            images_train, 
            images_val, 
            labels_train, 
            labels_val, 
            DEFAULT_PATH, 
            split_nr
            )
    print('Done')





    
