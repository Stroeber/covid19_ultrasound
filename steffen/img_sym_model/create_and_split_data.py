import os 
import pandas as pd
import numpy as np
from utils_data import delete_img_data_folder, create_images_from_videos, group_by_score, cross_val_split, store_split


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    DEFAULT_PATH = os.getcwd()
    os.chdir('data')

    metadata_raw = pd.read_csv('dataset_metadata.csv')

    relevant_columns = ['Current location','Filename', 'Label', 'Lung Severity Score', 'Type',
                        # 'Gender', 'Age', 
                        'Healthy', 'Fever', 'Cough', 'Respiratory problems', 
                        'Headache', 'Loss of smell/taste', 'Fatigue', 'Sore throat', 'Asymptomatic'
                        ]

    metadata = metadata_raw.filter(items=relevant_columns)
    metadata = metadata.replace('n/A', np.nan)
    for symptom in ['Fever', 'Cough', 'Respiratory problems', 
                    'Headache', 'Loss of smell/taste', 'Fatigue', 'Sore throat', 
                    'Asymptomatic', 'Healthy']:
                    metadata[symptom] = metadata[symptom].replace('0', -1).replace(np.nan, 0)
    metadata = metadata.dropna(subset=['Lung Severity Score',
                    #  'Age', 'Gender', 
                     'Fever', 'Cough', 'Respiratory problems', 
                    'Headache', 'Loss of smell/taste', 'Fatigue', 'Sore throat', 
                    'Asymptomatic', 'Healthy'
                    ])
    print(f'length: {len(metadata.index)}')

    grouped_images, grouped_scores, grouped_symptoms = create_images_from_videos(DEFAULT_PATH, metadata)

    scorelist = group_by_score(grouped_images, grouped_scores, grouped_symptoms)

    delete_img_data_folder(DEFAULT_PATH)

    # print((scorelist[3][1]))
    # exit()

    print('\nStart storing splits:')
    for score_nr in range(len(scorelist)):
        print(score_nr)
        
        indices = np.arange(len(scorelist[score_nr][0]))
        np.random.shuffle(indices)
        indices_val = np.array_split(indices, 5)

        for split_nr in range(len(indices_val)):
            (images_train, images_val, 
            labels_train, labels_val, 
            symptoms_train, symptoms_val) = cross_val_split(
                indices, 
                indices_val[split_nr], 
                scorelist[score_nr][0], 
                scorelist[score_nr][1], 
                scorelist[score_nr][2], 
                split_nr)

            store_split(
                images_train, images_val, 
                labels_train, labels_val, 
                symptoms_train, symptoms_val,
                DEFAULT_PATH, split_nr
            )
    print('Done')