"""
Author: Dhruvil A DARJI
created: March 19, 2018 6:55 PM
Used Method: Convolutoin Neural Networks
Library: Scikit-neuralnetwork 
python testing.py -i PATH TO THE FOLDER CONTAIN IMAGES AND FILES
"""
import cv2                      #import OPENCV to manipulate with images
import numpy as np              #numpy 
import argparse                 #it uses to pass the arguments from Commandline
import os                       # Operating System
import math                 
import csv                      #to read the file
import pickle                   #to store the data of the trained network. It can be useful with testing data
import sys

path = os.path.abspath(sys.argv[1][2:])
filename = 'pickle_file.pkl'    #'ready_pickle.pkl'  -Already Trained File
dim = (490.0,326.0)             #for normalize purpose
OVERLAP = 1.0                   #Overlap between two consecutive cutted images (1 means No overlap) (2 means half overlap) ...
cropped_img_dim = (36,48)       #Same as cutted in the Training

def read_images(path):  #Read Image
    aux = cv2.imread(path)
    aux = cv2.cvtColor(aux , cv2.COLOR_RGB2GRAY )
    images = aux
    return images

def Data_Augmentation(image, cropped_img_dim, OVERLAP):  #Classify images with the phone label.
    image_dim = (len(image),len(image[0]))                              #Image dimention
    vert = image_dim[0]/(cropped_img_dim[0]/OVERLAP)                            #Number of images make vertically out of one image 
    hori = image_dim[1]/(cropped_img_dim[1]/OVERLAP)                            #Number of images make vertically out of one image
    images = []
    conversion = []
    centers = []             
    for l in range(int(vert)):                                #Cut the image Vertically
        for k in range(int(hori)):                            #Cut the image Horizontally
            for i in range(int((cropped_img_dim[0]/OVERLAP)*l),int(cropped_img_dim[0]*((l/OVERLAP)+1))):        #Make new images with fixed height and width (cropped_omg_dim)
                for j in range(int((cropped_img_dim[1]/OVERLAP)*k),int(cropped_img_dim[1]*((k/OVERLAP)+1))): 
                    conversion.append(image[i][j])                     
            img = np.reshape(conversion,(cropped_img_dim))                                                      #One patch of an image (Cutted image)
            images.append(img)                                                                                 #Build train_x
            conversion = []
            centers.append((((cropped_img_dim[1]/OVERLAP)*(k+1))/dim[0],((cropped_img_dim[0]/OVERLAP)*(l+1))/dim[1]))  #Find_center of the image and normalise it
    images = np.array(images)
    return images,centers

image = read_images(path)                                     #Read images 
images, centers = Data_Augmentation(image, cropped_img_dim, OVERLAP)         #Label the data (Classify generated images from given images with 1's and 0's)
loaded_model = pickle.load(open(filename, 'rb'))                             #Load the pickle file 
prediction = loaded_model.predict(images)                                    #Predict an image with the phone
prediction_prob =  loaded_model.predict_proba(images)                        #Predict the probability of an image being a phone
pred = []                                                                           
for i in range(len(prediction)):
    if prediction[i] == [1]:                                                
        pred.append(prediction_prob[i][1])
    else:
        for i in range(len(prediction_prob)):
            pred.append(prediction_prob[i][1])
    index = pred.index(max(pred))
print round(centers[index][0],4),round(centers[index][1],4)                 #print the coordinates of the phone


