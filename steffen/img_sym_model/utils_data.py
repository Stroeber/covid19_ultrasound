import numpy as np
import re
import os
import cv2
from pathlib import Path
from shutil import rmtree
from tqdm import tqdm

def equal(a, b):
    # Ignore underscores and minus signs
    regex = re.compile(r'[_-]')
    return regex.sub('', a) == regex.sub('', b)


def flatten(xss):
    #Flatten a list of lists
    return np.array([x for xs in xss for x in xs], dtype=object)


def delete_img_data_folder(DEFAULT_PATH):
    path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_sym_data')
    if os.path.exists(path):
        rmtree(path)
        print('\nimg_sym_data folder has been deleted and a new random split will be created')


def create_images_from_videos(DEFAULT_PATH, metadata):

    grouped_images = []
    grouped_scores = []
    grouped_symptoms = []
    agg_nr_selected = []

    for (curr_loc, filename, sev_score, 
    # age, gender,
     healthy, fever, cough, 
    respiratory_problems, headache, loss_of_smell_taste, 
    fatigue, sore_throat, asymptomatic) in tqdm(zip(
        metadata['Current location'], 
        metadata['Filename'],
        metadata['Lung Severity Score'],
        # metadata['Age'],
        # metadata['Gender'],
        metadata['Healthy'],
        metadata['Fever'],
        metadata['Cough'],
        metadata['Respiratory problems'],
        metadata['Headache'],
        metadata['Loss of smell/taste'],
        metadata['Fatigue'],
        metadata['Sore throat'],
        metadata['Asymptomatic'],
        ),
        total=len(metadata['Current location']), 
        desc='Extracting images from videos', 
        ascii=True
        ):

        if curr_loc == 'butterfly':
            curr_loc = os.path.join('data', 'pocus_videos', 'convex')
        if curr_loc == 'not used':
            curr_loc = os.path.join('data', 'pocus_videos', 'convex')
        if not curr_loc.startswith('data'):
            curr_loc = os.path.join('data',  curr_loc)

        dir_path = os.path.join(DEFAULT_PATH, curr_loc)

        
        for video in os.listdir(dir_path):
            if equal(video.split('.')[0], filename):
                images = []
                scores = []
                symptoms = []

                video_path = os.path.join(dir_path, video)

                cap = cv2.VideoCapture(video_path)
                nr_selected = 0
                while cap.isOpened() and nr_selected < 30:
                    frameId = cap.get(1)
                    success, frame = cap.read()
                    if (success != True):
                        break
                    if frameId % 3 == 0:
                        images.append(frame)
                        scores.append(sev_score)
                        symptoms.append([
                                        # age, gender, 
                                        healthy, fever, cough, 
                                        respiratory_problems, headache, loss_of_smell_taste, 
                                        fatigue, sore_throat, asymptomatic])
                        nr_selected += 1
                            
                grouped_images.append(images)
                grouped_scores.append(scores)
                grouped_symptoms.append(symptoms)
                cap.release()
                agg_nr_selected.append(nr_selected)
    print(f'Got an average of {np.mean(agg_nr_selected)} images per video')
    return grouped_images, grouped_scores, grouped_symptoms

def cross_val_split(indices, indices_val, images, labels, symptoms, split_nr):

    images = np.array(images, dtype=object)
    labels = np.array(labels, dtype=object)
    symptoms = np.array(symptoms, dtype=object)

    indices_train = list(set(indices) - set(indices_val))
    images_train = flatten(images[indices_train])
    images_val = flatten(images[indices_val])
    labels_train = flatten(labels[indices_train])
    labels_val = flatten(labels[indices_val])
    symptoms_train = flatten(labels[indices_train])
    symptoms_val = flatten(labels[indices_val])

    print(f'\nsplit {split_nr}/4:')
    print(f'    Total images: {len(images_train) + len(images_val)}')
    print(f'    Training images: {len(images_train)}')
    print(f'    Validation Images: {len(images_val)}')

    return images_train, images_val, labels_train, labels_val, symptoms_train, symptoms_val


def store_split(images_train, images_val, 
                labels_train, labels_val, 
                symptoms_train, symptoms_val, 
                DEFAULT_PATH, split_nr
                ):

    split_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_sym_data', 'split' + str(split_nr))
    Path(split_path).mkdir(parents=True, exist_ok=True)

    for img, label, symptoms, id in tqdm(zip(
                                    images_train,  
                                    labels_train,
                                    symptoms_train, 
                                    range(len(images_train))), 
                                total=len(images_train), 
                                desc='Storing training images',
                                ascii=True
                                ):
        img_path = os.path.join(split_path, 'train', 'score' + str(int(label)))
        Path(img_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(img_path, str(id) + '.npz')
        np.savez(filename, image=img, symptoms=symptoms)


    for img, label, symptoms, id in tqdm(zip(
                                    images_val, 
                                    labels_val, 
                                    symptoms_val,
                                    range(len(images_val))),
                                total=len(images_val),
                                desc='Storing validation images',
                                ascii=True
                                ):
        img_path = os.path.join(split_path, 'validation', 'score' + str(int(label)))
        Path(img_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(img_path, str(id) + '.npz')
        np.savez(filename, image=img, symptoms=symptoms)