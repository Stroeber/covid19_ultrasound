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
    path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_data')
    if os.path.exists(path):
        rmtree(path)
        print('\nimg_data folder has been deleted and a new random split will be created')


def create_images_from_videos(DEFAULT_PATH, metadata):

    grouped_images = []
    grouped_scores = []
    agg_nr_selected = []

    for curr_loc, filename, sev_score in tqdm(
        zip(
            metadata['Current location'], 
            metadata['Filename'],
            metadata['Lung Severity Score']
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
                        nr_selected += 1
                            
                grouped_images.append(images)
                grouped_scores.append(scores)
                cap.release()
                agg_nr_selected.append(nr_selected)
    print(f'Got an average of {np.mean(agg_nr_selected)} images per video')
    return grouped_images, grouped_scores


def cross_val_split(indices, indices_val, images, labels, split_nr):

    images = np.array(images, dtype=object)
    labels = np.array(labels, dtype=object)

    indices_train = list(set(indices) - set(indices_val))

    images_train = flatten(images[indices_train])
    images_val = flatten(images[indices_val])
    
    labels_train = flatten(labels[indices_train])
    labels_val = flatten(labels[indices_val])

    print(f'\nsplit {split_nr}/4:')
    print(f'    Total images: {len(images_train) + len(images_val)}')
    print(f'    Training images: {len(images_train)}')
    print(f'    Validation Images: {len(images_val)}')

    return images_train, images_val, labels_train, labels_val


def store_split(images_train, images_val, labels_train, labels_val, DEFAULT_PATH, split_nr):

    split_path = os.path.join(DEFAULT_PATH, 'steffen', 'data', 'img_data', 'split' + str(split_nr))
    Path(split_path).mkdir(parents=True, exist_ok=True)

    for img, label, img_nr in tqdm(zip(
                                    images_train,  
                                    labels_train, 
                                    range(len(images_train))), 
                                total=len(images_train), 
                                desc='Storing training images',
                                ascii=True
                                ):
        img_path = os.path.join(split_path, 'train', 'score' + str(int(label)))
        Path(img_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(img_path, f'image{img_nr}.jpg')
        status  = cv2.imwrite(filename, img)
        if status == False:
            print(f"Couldn't write image{img_nr}.jpg to " + img_path)

    for img, label, img_nr in tqdm(zip(
                                    images_val, 
                                    labels_val, 
                                    range(len(images_val))),
                                total=len(images_val),
                                desc='Storing validation images',
                                ascii=True
                                ):
        img_path = os.path.join(split_path, 'validation', 'score' + str(int(label)))
        Path(img_path).mkdir(parents=True, exist_ok=True)

        filename = os.path.join(img_path, f'image{img_nr}.jpg')
        status  = cv2.imwrite(filename, img)
        if status == False:
            print(f"Couldn't write image{img_nr}.jpg to " + img_path)
