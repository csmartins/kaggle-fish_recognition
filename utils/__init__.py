from hist import *
from image_crop import *
from pca_zca_whitening import *
from classify import *

OUTPUT_DIR = 'cropped_train/'


def get_all_images_path_train():
    images_by_label = {}
    for folder in os.listdir(OUTPUT_DIR):
        folder_path = OUTPUT_DIR + folder
        images_by_label[folder] = []
        for img in os.listdir(folder_path):
            img_path = folder_path + '/' + img
            images_by_label[folder].append(img_path)
            
    return images_by_label

'''Para todas as pastas em cropped_train ira gerar as imagens esbranquicadas.'''
def white_all_train_folders(showImages, resizeImages):
    images_by_label = get_all_images_path_train()
    zcas = {}
    for folder in os.listdir(OUTPUT_DIR):
        print("\nWhitening " + folder)
        zca = whiten_images(images_by_label[folder], showImages, resizeImages)
        zcas[folder] = zca
    return zcas

def hog_whitened_images(images_whitened):
    images_hogged = {}
    for folder in images_whitened.keys():
        images_hogged[folder] = []
        print "Creating hog for ", folder
        for image in images_whitened[folder]:
            image_descriptor = generate_hog_descriptor(image)
            images_hogged[folder].append(image_descriptor)
    return images_hogged
    
def hog_all_images():
    images_by_label = get_all_images_path_train()
    images_hogged = {}    
    for folder in os.listdir(OUTPUT_DIR):
        images_hogged[folder] = []
        print "Creating hog for ", folder
        for image in images_by_label[folder]:
            image_vector = generate_hog_vector(image)
            #soma = sum(image_vector)
            #normalized = np.asarray([i/soma for i in image_vector])
            #normalized = normalize(image_vector)
            images_hogged[folder].append(image_vector)
    return images_hogged
