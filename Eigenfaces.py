# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:34:45 2017

@author: Abdullah Mobeen

This code displays one powerful application of Linear Algebra ~ Eigenfaces. 
It works on the Olivetti face dataset that contains 400 images of 64 x 64 pixels
The objective of this program is to reduce the dimensionality of every image
using a statistical technique called Principal Component Analysis. This will result in
Principal Components/Eigenfaces/Eigenvectors that can best approximate any face. This means
that a original face could be projected upon this new set of Eigenfaces (dimension < original dimension)
and still look very similar to the original one. We could also say that we get the 
basis for every face in our dataset. This techique is an essential idea behind 
Face Recognition.
"""

import numpy as np
import matplotlib.pyplot as plt
import random


def get_image_matrix(file):
    """Reads the file containing 400 images defined by their pixels and 
        returns a matrix containing 400 arrays where each array corresponds
        to each image. Each value in the arrays is the pixel value on a greyscale (0-255)"""
    data = np.genfromtxt(file, dtype=int, delimiter = ',')
    return data

def first_face(data):
    """Function that displays the first face from the dataset.
    Takes as input:
            data = the matrix containing all the face image vectors"""
    global image_count
    first_face = np.reshape(data[0],(64,64),order='F')
    
    image_count += 1    
    plt.figure(image_count)
    plt.title('First_face')
    plt.imshow(first_face,cmap=plt.cm.gray)


def random_face(data):
    """Function that displays a random face from the dataset.
    Takes as inputs:
            data = the matrix containing all the face image vectors"""
    global image_count
    rand_int = random.randint(0,len(data))
    rand_face = np.reshape(data[rand_int],(64,64), order = 'F')
    
    image_count += 1
    plt.figure(image_count)
    plt.title('Random Face')
    plt.imshow(rand_face, cmap=plt.cm.gray)


def mean_face(data):
    """Function that computes and returns the mean face of all the faces in the data
    Takes as input:
            data = the matrix containing all the face image vectors"""
    mean_face = np.zeros((len(data[0]),), dtype = int)
    for i in range(len(data)):
        mean_face += data[i]
    
    mean_face = mean_face/len(data)
    return mean_face 

def mean_face_construct(data, mean_face):
    """Function that displays the mean face.
    Takes as inputs:
            data = the matrix containing all the face image vectors,
            mean_face = the mean face vector
    """
    global image_count
    disp_mean = np.reshape(mean_face,(64,64), order = 'F')
    
    image_count += 1    
    plt.figure(image_count)
    plt.title('Mean Face')
    plt.imshow(disp_mean, cmap=plt.cm.gray)
    
    
def centre_matrix(data):
    """Function that computes the matrix containing the centered face vectors i.e.
    face vector - mean face vector, for each face vector in the original matrix
    Takes as inputs:
            data = the matrix containing all the face image vectors"""
    matrix = []
    for i in range(len(data)):
        v = data[i] - mean_face
        matrix.append(v)
    matrix = np.matrix(matrix)
    return matrix

def covariance_matrix(matrix):
    """Function that computes a matrix L such that L = (A.A(transpose))
    This is a computational trick which results in a matrix of 400 x 400 dimensions
    as compared to (A(transpose).A), which would result in a matrix of
    4096 x 4096 dimensions. It then computes and returns the eigenvalues and eigenvectors of
    this matrix L, which both could be used then to compute the eigenvectors of A.
    Takes as inputs:
            matrix = matrix of 'centered face vectors', i.e. face vector - mean face vector
    """
    L = np.matmul(matrix, np.transpose(matrix))
    L_eigenvalues, L_eigenvectors = np.linalg.eigh(L)
    
    idx = np.argsort(-L_eigenvalues)
    L_eigenvalues = L_eigenvalues[idx]
    L_eigenvectors = L_eigenvectors[:,idx]
    
    return L_eigenvalues, L_eigenvectors


def A_eigenvectors(matrix, L_eigenvectors, L_eigenvalues):
    """Function that computes the the eigenvectors of the matrix containing all the
    images. Orders the eigenvectors in the decreasing order of eigenvalues. 
    Return the matrix containing these ordered eigenvectors.
    Takes as input:
            matrix = matrix of 'centered face vectors', i.e. face vector - mean face vector,
            L_eigenvectors = eigenvectors of the matrix L = (A.A(transpose))
            L_eigenvalues = eigenvalues of the matrix L = (A.A(transpose))
    """
    
    A_eigenvectors = np.dot(matrix.T, L_eigenvectors)
    
    idx = np.argsort(-L_eigenvalues)
    L_eigenvalues = L_eigenvalues[idx]
    A_eigenvectors = A_eigenvectors[:,idx]
    
    for i in range(len(L_eigenvectors)):
        A_eigenvectors[:,i] = A_eigenvectors[:,i]/np.linalg.norm(A_eigenvectors[:,i])

    return A_eigenvectors
    
def PC_10(A_eigenvectors):
    """Function that returns the 10 eigenvectors corresponding to 10 eigenvectors
    with the maximum value in descending order.
    Takes as input:
            A_eigenvectors = list of eigenvectors corresponding to eigenalues
                             in descending order i.e. at index 0 = eigenvector
                             corresponding to the largest eigenvalue"""
    
    PC_10 = A_eigenvectors[:,:10]
    return PC_10
    

def PC2_construct(data, mean_face, A_eigenvectors):
    """Function that reconstructs the first face by projecting it on 
    2 PCs. Takes as inputs:
                    data = matrix containing the pictures,
                    mean_face = the mean face vector of all the image vectors,
                    A_eigenvectors = list of eigenvectors corresponding to eigenalues
                                     in descending order i.e. at index 0 = eigenvector
                                     corresponding to the largest eigenvalue"""
    global image_count
    f_face = data[0]
    mean_f = f_face - mean_face
    
    PC_2 = A_eigenvectors[:,:2]
    ohm_2 = np.dot(np.transpose(PC_2),(mean_f))
    mean_face = np.reshape(mean_face,(len(data[0]),1), order = 'F')
    proj_2 = np.add(mean_face, np.dot(PC_2,np.transpose(ohm_2)))
    
    proj_face_2 = np.reshape(proj_2,(64,64), order = 'F')
    
    image_count += 1
    plt.figure(image_count)
    plt.title('2 Principal Components')
    plt.imshow(proj_face_2, cmap=plt.cm.gray)


def PC_construct(data, p, A_eigenvectors,mean_face, rand):
    """Function that reconstructs a random face using the p number of 
    PCs. Takes as inputs:
                data = matrix containing the pictures,
                p = number of Principal Components (eigenvectors) the image is projected upon,
                A_eigenvectors = list of eigenvectors corresponding to eigenalues
                                 in descending order i.e. at index 0 = eigenvector
                                 corresponding to the largest eigenvalue,
                mean_face = the mean face vector of all the image vectors
                rand = random integer that is used to index a random face"""
                    
    global image_count
    rand_face = data[rand]
    mean_f = rand_face - mean_face
    
    PC = A_eigenvectors[:,:p]
    ohm_p = np.dot(np.transpose(PC),(mean_f))
    mean_face = np.reshape(mean_face,(len(data[0]),1), order = 'F')
    proj_p = np.add(mean_face, np.dot(PC,np.transpose(ohm_p)))
    proj_face_p = np.reshape(proj_p,(64,64), order = 'F')
    
    image_count += 1   
    plt.figure(image_count)
    plt.title('K Principal Components')
    plt.imshow(proj_face_p, cmap=plt.cm.gray)



if __name__ == "__main__":
    image_count = 0
    data = get_image_matrix('faces.csv')
    first_face(data)
    random_face(data)
    mean_face = mean_face(data)
    
    mean_face_construct(data, mean_face)
    
    C = centre_matrix(data)
    L_eigenvalues, L_eigenvectors = covariance_matrix(C)
    A_eigenvectors = A_eigenvectors(C, L_eigenvectors, L_eigenvalues)
    
    PC_10 = PC_10(A_eigenvectors)
    
    PC2_construct(data,mean_face, A_eigenvectors)
    
    rand = random.randint(0,len(data))
    for i in [5,10,25,50,100,200,300,399]:
        PC_construct(data, i, A_eigenvectors,mean_face, rand)