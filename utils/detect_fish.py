import os
import numpy as np
import json
import cv2
import time
from progressbar import *
import random
import os.path as P
import os
import glob

'''PARA TESTE === TEMPORARIO'''
OUTPUT_DIR = '../../train/'
path =OUTPUT_DIR + os.listdir(OUTPUT_DIR)[0] 
img_test = path + '/' + os.listdir(path)[0]

minWidthPyramid = 64
minHeightPyramid = 64

'''Funcao responsavel por rotacionar uma imagem ao redor do centro da mesma, sem escalas.
Recebe como parametro a imagem e o angulo de rotacao.'''
def rotate_image(img, angle, extra_coords):
    rows,cols = img.shape[0:2]
    assert rows == cols, "img should be a square matrix"

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    new_coords = []
    for coords in extra_coords:
        ones = np.ones(shape=(coords.shape[0], 1))
        points_ones = np.hstack([coords, ones])
        transformed = M.dot(points_ones.T).T.astype('int')

        accum = angle
        while accum > 0:
            c1 = transformed[2, :]
            c2 = transformed[0, :]
            c3 = transformed[3, :]
            c4 = transformed[1, :]
            transformed = np.vstack([c1, c2, c3, c4])
            accum -= 90

        new_coords.append(transformed)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst, new_coords

'''Funcao responsavel por gerar a piramide da imagem e todas as suas rotacoes (90,180,270).
Recebe como parametro a imagem, uma escala que define o quanto uma imagem sera menor que a anterior (iniciando
com a imagem original) e o minSize que sera o tamanho minimo e criterio de parada.'''
def pyramid(image, scale, minSize, extra_coords=[]):
    yield image, extra_coords
    yield rotate_image(image, 90, extra_coords=extra_coords)
    yield rotate_image(image, 180, extra_coords=extra_coords)
    yield rotate_image(image, 270, extra_coords=extra_coords)

    while True:
        old_size = image.shape
        new_size = int(image.shape[0] * scale), int(image.shape[1] * scale)
        image = cv2.resize(image, new_size)

        new_coords = []
        for coords in extra_coords:
            coords[:, 0] = (coords[:, 0] * scale).astype('int')
            coords[:, 1] = (coords[:, 1] * scale).astype('int')
            new_coords.append(coords)
        extra_coords = new_coords
 
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
             
        yield image, extra_coords
        yield rotate_image(image,90, extra_coords=extra_coords)
        yield rotate_image(image,180, extra_coords=extra_coords)
        yield rotate_image(image,270, extra_coords=extra_coords)

'''Funcao que retorna as imagens da piramide e rotacao da imagem parametro.'''
def image_pyramid(img, showImage):
    image = cv2.imread(img)
    if image is None:
        raise ValueError('%s not found' % img)
    image = cv2.resize(image, (256, 256)) 
    yield image
    
    for (i, resized) in enumerate(pyramid(image, 0.7, (minWidthPyramid,minHeightPyramid))):
        if showImage:
            cv2.imshow("Layer {}".format(i + 1), resized)
            cv2.waitKey(0)
        yield resized
    
    if showImage:
        cv2.destroyAllWindows()
        
    
'''Funcao responsavel por criar janelas que percorrerao toda a imagem (sliding).
Recebe como parametro a imagem, o stepSize que dira quantos pixels deverao ser movidos
de uma janela para a proxima e windowSize que dira o tamanho das janelas.
Retorna um enumerate com a posicao da janela e a imagem resultado.'''
def sliding_window(image, stepSize, windowSize):
    #y vai de zero ate a altura da imagem com o passo de tamanho stepSize
    for y in range(0, image.shape[0] - windowSize[1], stepSize):
        #x vai de zero a largura da imagem com o passo de tamanho stepSize
        for x in range(0, image.shape[1] - windowSize[0], stepSize):
            # retorna largura e altura e imagem
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
            
'''Funcao responsavel por criar o dataset de treino a partir das sliding windows das imagens 
provenientes do pyramid das imagens originais.
Usa o criterio de overlapping e as anotacoes de posicoes dos peixes (no csv fish_annotation) para
determinar se a window possui peixe ou nao. Salva todas as windows e um csv com o nome de cada uma 
e seu resultado do overlapping com o peixe real.'''
def collect_train_data(images_dir, positions_dir, output_dir):
    from clint.textui.progress import bar

    positions_files = glob.glob(P.join(positions_dir, '*.json'))

    if not P.isdir(output_dir):
        os.makedirs(output_dir)

    csv_file = open('%s.csv' % output_dir, 'w')

    count = 0
    for f in positions_files:
        class_name = P.basename(f).upper().split('.')[0]
        print('CLASS - %s' % class_name)

        with open(f) as fin:
            data = json.load(fin)
            for img_data in bar(data):
                img_file = P.join(images_dir, class_name, img_data['filename'])

                boxes = []
                for ann in img_data['annotations']:
                    ulx = ann['x']
                    uly = ann['y']

                    blx = ulx
                    bly = uly + ann['height']

                    urx = ulx + ann['width']
                    ury = uly

                    brx = urx
                    bry = bly

                    boxes.append(np.asarray([
                        [ulx, uly],
                        [blx, bly],
                        [urx, ury],
                        [brx, bry]
                    ]))
                img = cv2.imread(img_file)
                if img is None:
                    raise ValueError('%s not found' % img_file)

                original_shape = img.shape
                img = cv2.resize(img, (512, 512))
                new_shape = img.shape
                new_boxes = []
                for box in boxes:
                    box[:, 0] = (box[:, 0] * (new_shape[1] / original_shape[1])).astype('int')
                    box[:, 1] = (box[:, 1] * (new_shape[0] / original_shape[0])).astype('int')
                    new_boxes.append(box)
                boxes = new_boxes

                for scaled, scaled_boxes in pyramid(img, 0.7, (64, 64), boxes):
                    for x, y, window in sliding_window(scaled, 16, (32, 32)):
                        ulx = x
                        uly = y

                        blx = x
                        bly = y + 32

                        urx = x + 32
                        ury = y

                        brx = urx
                        bry = bly

                        window_box = np.asarray([
                            [ulx, uly],
                            [blx, bly],
                            [urx, ury],
                            [brx, bry]
                        ])

                        overlap = False
                        for fish_box in scaled_boxes:
                            if is_overlapping(window_box, fish_box, factor=0.7):
                                overlap = True
                                break
                        if not overlap and random.random() < 0.99:
                            continue
                        out_im_name = '%09d.png' % count
                        count += 1
                        csv_file.write('%s,%s,%s\n' % (img_file, out_im_name, overlap))
                        cv2.imwrite(P.join(output_dir, out_im_name), window)

"""Le as imagens das janelas, aplica whitening e hog e retorna as matrizes x e y
para qualquer algoritmo de classificacao usa-las."""
def read_process_data(images_dir, train_samples=10000, discard_false=0.9):
    from skimage.feature import hog
    from skimage import color
    from skimage.io import imread
    from clint.textui.progress import bar
    from pca_zca_whitening import whiten_image

    csv_file = open('%s.csv' % images_dir)

    x = np.empty(shape=(train_samples, 2048))
    y = np.empty(shape=(train_samples,), dtype='int8')
    count = 0
    for i, line in bar(enumerate(csv_file), expected_size=train_samples*10):
        if count == train_samples:
            break
        line = line.rstrip()
        tokens = line.split(',')
        img_file = tokens[1]
        answer = tokens[2]
        if answer == 'True':
            answer = 1
        else:
            if random.random() < discard_false:
                continue
            answer = 0
        img = whiten_image(P.join(images_dir, img_file))
        fd = hog(img, orientations=8, pixels_per_cell=(2, 2),
                 cells_per_block=(1, 1))
        x[count] = fd
        y[count] = answer
        count += 1
    if count < train_samples:
        raise ValueError("Didn't find %d samples to read" % train_samples)

    return x, y

'''Realiza o svm nas matrizes x e y passadas como parametro, utilizando os parametros
C e kernel para o svm. Retorna uma matriz de confusao para os resultados obtidos.'''
def evaluate_svm(x, y, test_split, C=1, kernel='rbf'):
    from sklearn.model_selection import train_test_split

    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=test_split)

    from sklearn.svm import SVC

    classifier = SVC(C=C, kernel=kernel)
    classifier.fit(x_train, y_train)
    r = classifier.predict(x_valid)

    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_valid, r)
    
"""Checa se duas boxes estao sobrepostas pelo fator factor do parametro.
Cada box eh uma matriz com 4 linhas e 2 colunas, representando os pontos, em ordem:
    - esquerda superior
    - esquerda inferior
    - direita superior
    - direita inferior
    
Recebe como parametro box1 e box2 como sendo um array de pontos (x,y) de suas delimitacoes, 
e factor como o fator de sobreposicao entre as duas caixas."""
def is_overlapping(box1, box2, factor=0.8):
    box1 = np.asarray(box1)
    box2 = np.asarray(box2)

    height1 = box1[1, 1] - box1[0, 1]
    width1 = box1[2, 0] - box1[0, 0]
    area1 = height1 * width1

    height2 = box2[1, 1] - box2[0, 1]
    width2 = box2[2, 0] - box2[0, 0]
    area2 = height2 * width2

    if area1 < area2:
        bigger = area2
    else:
        bigger = area1

    ys = [(1, box1[0, 1]), (1, box1[1, 1]),
          (2, box2[0, 1]), (2, box2[1, 1])]
    ys.sort(key=lambda y: y[1])

    order = list(map(lambda y: y[0], ys))
    if order == [1, 1, 2, 2] or order == [2, 2, 1, 1]:
        return False
    overlap_y = abs(ys[1][1] - ys[2][1])

    xs = [(1, box1[0, 0]), (1, box1[2, 0]),
          (2, box2[0, 0]), (2, box2[2, 0])]
    xs.sort(key=lambda x: x[1])

    order = list(map(lambda x: x[0], xs))
    if order == [1, 1, 2, 2] or order == [2, 2, 1, 1]:
        return False
    overlap_x = abs(xs[1][1] - xs[2][1])

    overlap_area = overlap_x * overlap_y

    return (overlap_area / bigger) >= factor


'''========================================= FUNCOES DE TESTE PARA VER FUNCIONAMENTO =================================================='''
def test_pyramid():
    print("Pyramid of " + img_test)
    list(image_pyramid(img_test, True))
    
def test_sliding():
    image = cv2.imread(img_test)
    if image is None:
        raise ValueError('%s not found' % img_test)
    image = cv2.resize(image, (256, 256)) 
    winH= 32
    winW = 32
    extra_coords = [
        np.asarray([[100, 100]])
    ]
    for resized, coords in pyramid(image, 0.7, (64,64), extra_coords=extra_coords):
        for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            cv2.circle(clone, tuple(coords[0][0]), 4, (255,0,0), 3)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            time.sleep(0.05)
    
'''Funcao de teste para leitura de todos os arquivos do dataset de treino.'''
def test_read_images():
    
    images = []
    DIR = '../../train/'
    for folder in os.listdir(DIR):
        if folder == 'NoF':
            pass
            
        print('reading samples '+folder)
        path = DIR + folder 
        
        total = 0
        progBar = ProgressBar(len(os.listdir(path)))
        progBar.start()
        for img in os.listdir(path):
            img_read = path + '/' + img
            images.append(cv2.imread(img_read))
            total = total + 1
            progBar.update(total)
        progBar.finish()
                
if __name__ == '__main__':
    #test_pyramid()
    #test_sliding()
    #test_read_images()
    ...
