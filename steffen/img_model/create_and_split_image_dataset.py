import os
import pandas as pd
import numpy as np
from utils_data import (
    delete_img_data_folder, 
    create_images_from_videos, 
    cross_val_split, 
    store_split, 
    group_by_score
)


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

    scorelist = group_by_score(grouped_images, grouped_scores)

    delete_img_data_folder(DEFAULT_PATH)

    print('\nStart storing splits:')
    for score_nr in range(len(scorelist)):
        print(score_nr)
        
        indices = np.arange(len(scorelist[score_nr][0]))
        np.random.shuffle(indices)
        indices_val = np.array_split(indices, 5)

        for split_nr in range(len(indices_val)):
            (images_train, images_val, 
            labels_train, labels_val) = cross_val_split(
                indices, 
                indices_val[split_nr], 
                scorelist[score_nr][0], 
                scorelist[score_nr][1], 
                split_nr)

            store_split(
                images_train, images_val, 
                labels_train, labels_val, 
                DEFAULT_PATH, split_nr
            )
    print('Done')





    
