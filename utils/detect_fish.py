import os
import cv2
import imutils
import time
from progressbar import *
'''PARA TESTE === TEMPORARIO'''
OUTPUT_DIR = '../../train/'
path =OUTPUT_DIR + os.listdir(OUTPUT_DIR)[0] 
img_test = path + '/' + os.listdir(path)[0]

minWidthPyramid = 64
minHeightPyramid = 64

'''Funcao responsavel por rotacionar uma imagem ao redor do centro da mesma, sem escalas.
Recebe como parametro a imagem e o angulo de rotacao.'''
def rotate_image(img, angle):
    rows,cols = img.shape[0:2]
    assert rows == cols, "img should be a square matrix"

    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst

'''Funcao responsavel por gerar a piramide da imagem. A piramide consiste em varias imagens provenientes
da original, porem com tamanho variados. Importante para quando usarmos o sliding window ser facil de reconhecer
peixes de tamanho variados.
Recebe como parametro a imagem, uma escala que define o quanto uma imagem sera menor que a anterior (iniciando
com a imagem original) e o minSize que sera o tamanho minimo e criterio de parada.'''
def pyramid(image, scale, minSize):
    yield image
    yield rotate_image(image,90)
    yield rotate_image(image,180)
    yield rotate_image(image,270)

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)
 
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break
             
        yield image
        yield rotate_image(image,90)
        yield rotate_image(image,180)
        yield rotate_image(image,270)
		
'''Funcao que retorna as imagens da piramide da imagem parametro.'''
def image_pyramid(img, showImage):
    image = cv2.imread(img)
    image = cv2.resize(image, (256, 256)) 
    yield image
    
    for (i, resized) in enumerate(pyramid(image, 1.3, (minWidthPyramid,minHeightPyramid))):
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
	for y in xrange(0, image.shape[0], stepSize):
	    #x vai de zero a largura da imagem com o passo de tamanho stepSize
		for x in xrange(0, image.shape[1], stepSize):
			# retorna largura e altura e imagem
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])
			
'''========================================= FUNCOES DE TESTE PARA VER FUNCIONAMENTO ==================================================
                                              SERAO APAGADAS FUTURAMENTE                                                                '''
def test_pyramid():
    print("Pyramid of " + img_test)
    list(image_pyramid(img_test, True))
    
def test_sliding():
    image = cv2.imread(img_test)
    image = cv2.resize(image, (256, 256)) 
    winH= 32
    winW = 32
    for resized in pyramid(image, 1.3, (64,64)):
    	for (x, y, window) in sliding_window(resized, stepSize=16, windowSize=(winW, winH)):
            if window.shape[0] != winH or window.shape[1] != winW:
                continue

            #INSERIR METODO PARA RECONHECER PEIXE

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            #cv2.imshow("Window", clone)
            #cv2.waitKey(1)
            #time.sleep(0.05)
            
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
    progbar = ProgressBar(1000)
    progbar.start()
    for i in range(1000):
        progbar.update(i)
        test_sliding()
    progbar.finish()
    #test_read_images()
