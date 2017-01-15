from utils import *


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    images_whitened = white_all_train_folders(False, False)
    images_hogged = hog_whitened_images(images_whitened)

    svms = create_exemplar_svms(images_hogged)

    print svms
    
