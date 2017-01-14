import os
from utils import *

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    #white_all_train_folders(False, False)
    images_hogged = hog_all_images()

    '''svms = {}
    for folder in os.listdir(OUTPUT_DIR):
        svms[folder] = []
        for image in images_hogged[folder]:
            clf = svm.SVC(gamma=0.001, C=100.)
            clf.fit(image, folder)
            svms[folder].append(clf)'''
    
