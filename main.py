import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from image_crop import *
from whiten_images import *
from hist import *
from sklearn.preprocessing import normalize
from sklearn import svm

OUTPUT_DIR = 'cropped_train/'

'''Para todas as pastas em cropped_train ira gerar as imagens esbranquicadas.'''
def white_all_folders():
    zcas = {}
    for folder in os.listdir(OUTPUT_DIR):
        print("\nWhitening " + folder)
        
        path = OUTPUT_DIR + folder
        imgs = []
        
        for image in os.listdir(path):
            img = cv2.imread(path+'/'+image)
            resized_image = cv2.resize(img, (128, 128)) 
            imgs.append(resized_image)
        
        matrix = images_matrix(imgs)
        zca = zca_whitening(matrix)
        zcas[folder] = zca
        '''plt.figure()
        plt.plot(zca[0,:], zca[1,:], 'o', mec='blue', mfc='none')
        plt.title('xZCAWhite')
        plt.show()'''

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
        
    #white_all_folders()
    images_hogged = hog_all_images()

    svms = {}
    for folder in os.listdir(OUTPUT_DIR):
        svms[folder] = []
        for image in images_hogged[folder]:
            clf = svm.SVC(gamma=0.001, C=100.)
            clf.fit(image, folder)
            svms[folder].append(clf)
    
