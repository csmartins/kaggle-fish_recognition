import os
import numpy as np
from sklearn import svm
from progressbar import *

OUTPUT_DIR = 'cropped_train/'
classes_sizes = {}

def select_same_size_images(folder_images):
    max_size = 0
    for image in folder_images:
        img_len = len(image)
        if img_len > max_size:
            max_size = img_len
    
    #print "final max_size", max_size

    sizes = np.zeros(max_size+1)
    for image in folder_images:
        img_len = len(image)
        sizes[img_len] += 1

    #print 'all sizes', sizes

    selected_size = sizes.tolist().index(max(sizes))
    selected_images = []
    for image in folder_images:
        if len(image) == selected_size:
            selected_images.append(image)

    return np.array(selected_images)

def create_exemplar_svms(images_hogged):
    svms = {}
    for folder in os.listdir(OUTPUT_DIR):
        print "Creating Exemplar-SVMs for", folder
        svms[folder] = []
        folder_images = np.array(images_hogged[folder])#, dtype=np.float)

        progBar = ProgressBar(len(folder_images))
        progBar.start()
        count = 0
        #selected_images = select_same_size_images(folder_images) 
        #classes_sizes[folder] = len(selected_images[0])
       
        for i in range(0, len(folder_images)):
            features = ['not-'+folder]*len(folder_images)
            features[i] = folder
            features = np.asarray(features)

            clf = svm.SVC(gamma=0.001, C=100., kernel='rbf')
            clf.fit(folder_images, features)

            svms[folder].append(clf)
            count = count+1
            progBar.update(count)
        progBar.finish()
    return svms

def divide_dataset(images):
    train = {}
    validation = {}
    #validation_classes = []
    for folder in images.keys():
        total_images = len(images[folder])
        train_percentage = int((70*total_images)/100)
        folder_images = images[folder]

        np.random.seed(42)
        np.random.shuffle(folder_images)

        train[folder] = folder_images[:train_percentage]
        validation[folder] = folder_images[train_percentage:]
        #validation_classes.extend(folder for i in folder_images[train_percentage:])

    return train, validation#, validation_classes
