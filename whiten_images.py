import numpy as np

'''Retorna um vetor flatten da imagem passada como parametro'''
def flatten_matrix(matrix):
    vector = matrix.flatten(1)
    vector = vector.reshape(1, len(vector))
    return vector[0]
    
'''Retorna uma matriz n x m*z, onde n eh o numero total de imagens, m e z sao as dimensoes das imagens'''
def images_matrix(images):
    matrix = []
    for image in images:
        flat = flatten_matrix(image)
        matrix.append(flat)
    return np.matrix(matrix)

'''Retorna as imagens depois de ocorrer o whitening'''
def zca_whitening(inputs):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening
