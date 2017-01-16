from utils import *


if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
        crop_train_dataset()
        
    images_whitened = white_all_train_folders(False, True)
    images_hogged = hog_whitened_images(images_whitened)

    min_max_scaler = preprocessing.MinMaxScaler()
    
    imgs_normalized = {}
    for folder in images_hogged.keys():
        images_normalized = min_max_scaler.fit_transform(images_hogged[folder])
        imgs_normalized[folder] = images_normalized

    train, validation = divide_dataset(imgs_normalized)

    svms = create_exemplar_svms(train)

    for folder in svms.keys():
        #selected_for_validation = []
        #selected_validation_classes = []
        #for i in range(0, len(validation)):
        #    if len(validation[i]) == classes_sizes[folder]:
        #        selected_for_validation.append(validation[i])
        #        selected_validation_classes.append(validation_classes[i])

        #print "Selected", len(selected_for_validation), "images for validation"
        #if len(selected_for_validation) > 0:

        print "Starting validation for", folder
        progBar = ProgressBar(len(folder_images))
        progBar.start()
        count = 0
        for svm in svms[folder]:
            for img in validation[folder]:
                predict = svm.predict([img])
                if predict == folder:
                    print "ACERTEI"
            count = count+1
            progBar.update(count)
        progBar.finish()
