import os
import cv2
import numpy as np
#import matplotlib.pyplot as plt
from image_crop import *
from whiten_images import *

OUTPUT_DIR = 'cropped_train/'

'''Para todas as pastas em cropped_train ira gerar as imagens esbranquicadas.'''
def white_all_folders():
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
        
        '''plt.figure()
        plt.plot(zca[0,:], zca[1,:], 'o', mec='blue', mfc='none')
        plt.title('xZCAWhite')
        plt.show()'''

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    white_all_folders()
        
