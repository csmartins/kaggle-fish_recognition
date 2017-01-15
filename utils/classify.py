import os
import numpy as np
from sklearn import svm

OUTPUT_DIR = 'cropped_train/'

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

        selected_folder_images = select_same_size_images(folder_images)        
        for i in range(0, len(selected_folder_images)):
            #print len(image)
            features = np.zeros(len(selected_folder_images))
            features[i] = 1
            clf = svm.SVC(gamma=0.001, C=100.)
            clf.fit(selected_folder_images, features)

            svms[folder].append(clf)

    return svms

