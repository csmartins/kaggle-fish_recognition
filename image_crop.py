import json
import glob
import cv2
from progressbar import *
import os

OUTPUT_DIR = 'cropped_train/'
POSITIONS_DIR = 'fish_positions/'
TRAIN_DIR = 'train/'

'''A partir do json contendo a posicao dos peixes nas imagens, cria uma nova pasta com as imagens cortadas apenas nos peixes'''
def crop_images(positions_file):
	file_name = os.path.basename(positions_file)
	class_name = file_name.upper().split('.')[0]
	if not os.path.isdir(OUTPUT_DIR + class_name):
		os.makedirs(OUTPUT_DIR + class_name)

	print("\nCropping " + file_name)

	with open(positions_file) as data_file:
		data = json.load(data_file)

		progBar = ProgressBar(len(data))
		count = 0
		progBar.start()
		for img_data in data:
			img_file = TRAIN_DIR + class_name + '/' + img_data['filename']

			count_fishes = 0
			for ann in img_data['annotations']:
				img = cv2.imread(img_file)
				x_left = max(0, ann['x'])
				y_up = max(0,ann['y'])
				height = max(0, ann['height'])
				width = max(0, ann['width'])
				x_left, y_up, height, width = int(x_left), int(y_up), int(height), int(width)
				img = img[y_up:y_up+height, x_left:x_left+width]
				cv2.imwrite(OUTPUT_DIR + class_name + '/' + (str(count_fishes)+ '_' if count_fishes == 0 else '') + img_data['filename'], img)
				count_fishes = count_fishes + 1
		
			count = count + 1
			progBar.update(count)
    
        progBar.finish()				


'''Para cada pasta do dataset de treino chama a funcao de recortar os peixes'''
def crop_train_dataset():
    positions_files = glob.glob(POSITIONS_DIR + '*.json')
    for position_file in positions_files:
        crop_images(position_file)

if __name__ == '__main__':
    if not os.path.isdir(OUTPUT_DIR):
		os.mkdir(OUTPUT_DIR)

    crop_train_dataset()
