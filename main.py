from utils import *


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    #white_all_train_folders(False, False)
    images_hogged = hog_all_images()

    svms = create_exemplar_svms(images_hogged)
    
