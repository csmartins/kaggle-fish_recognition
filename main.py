import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn import svm

from image_crop import *
from hist import *
from utils.pca_zca_whitening import *

OUTPUT_DIR = 'cropped_train/'

'''Para todas as pastas em cropped_train ira gerar as imagens esbranquicadas.'''
def white_all_train_folders():
    zcas = {}
    for folder in os.listdir(OUTPUT_DIR):
        print("\nWhitening " + folder)
        path = OUTPUT_DIR + folder
        imgs = [path + '/' + s for s in os.listdir(path)] 
        zca = whiten_images(imgs, True, False)
        zcas[folder] = zca
    return zcas

def hog_all_images():
    images_hogged = {}    
    for folder in os.listdir(OUTPUT_DIR):
        images_hogged[folder] = []
        print "Creating hog for ", folder
        for image in os.listdir(OUTPUT_DIR + '/' + folder):
            image_path = OUTPUT_DIR + '/' + folder + '/' + image
            image_vector = generate_hog_vector(image_path)
            #soma = sum(image_vector)
            #normalized = np.asarray([i/soma for i in image_vector])
            #normalized = normalize(image_vector)
            images_hogged[folder].append(image_vector)
    return images_hogged

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    white_all_train_folders()
    #images_hogged = hog_all_images()

    '''svms = {}
    for folder in os.listdir(OUTPUT_DIR):
        svms[folder] = []
        for image in images_hogged[folder]:
            clf = svm.SVC(gamma=0.001, C=100.)
            clf.fit(image, folder)
            svms[folder].append(clf)'''
    
