"""
Author: Dhruvil A DARJI
created: March 19, 2018 6:55 PM
Used Method: Convolutoin Neural Networks
Library: Scikit-neuralnetwork 
python trai_phone_finder.py ~/find_phone
"""
from sklearn import datasets, cross_validation
from sknn.mlp import Classifier, Layer, Convolution
import cv2                      #import OPENCV to manipulate with images
import numpy as np              #numpy 
import os                       # Operating System
import math                 
import csv                      #to read the file
import re
import pickle                   #to store the data of the trained network. It can be useful with testing data
import pathlib
import sys
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"  #to execute your script in GPU
import theano                                                           #It detects your GPU (Note: you may comment it if you don't want to use your GPU)
dim = (490.0,326.0)             #for normalize purpose
path = os.path.abspath(sys.argv[1][2:]) #read the argument
OVERLAP = 1.0                   #Overlap between two consecutive cutted images (1 means No overlap) (2 means half overlap)...
cropped_img_dim = (36,48)       #cut an image in several images with cropped image dimensions.
write_label_file_path = path+"\extractedLabel.txt"    #file will be generated to read proper labels imagewise.
num_of_duplicates = 30          #duplicate number of images containing phone
THRESHOLD = 0.05                #Threshold is given
def Label_Exraction(read_label_file_path,write_label_file_path):
    with open(read_label_file_path) as tsv:
        listLabels = []
        for line in csv.reader(tsv, dialect="excel-tab"):
            strLine = ''.join(line)
            lineExtracted = re.sub('.jpg','',strLine)
            listLabels.append(lineExtracted)
        sortedList = sorted(listLabels, key=lambda x: (int)(x.split(' ')[0]))
        outputFile = open(write_label_file_path, 'w')
        for item in sortedList:
            outputFile.write("%s\n" % item)

def Find_Distance(coordinate1,coordinate2):
    a = math.sqrt((coordinate1[0]-coordinate2[0])**2 + (coordinate1[1]-coordinate2[1])**2)
    return a

def Read_Given_Labels(filename,path):   #Read phone coordinates and image numbers 
    a = []
    phone_coordinates = {}
    fileNumList = []
    image_path_list = []  
    with open(filename) as tsv:
        for line in tsv.readlines():         
            phone_coordinates[int(line.split(' ')[0])] =  (float(line.split(' ')[1]) , float(line.split(' ')[2]))
    for file in os.listdir(path):
        extension = os.path.splitext(file)[1]
        if extension == ".jpg":
            image_path_list.append(os.path.join(path, file))
            temp = os.path.splitext(file)[0]
            fileNumList.append(int(temp))
    fileNumList =  np.sort(fileNumList)
    return phone_coordinates,fileNumList

def Read_Images(fileNumList,path):  #Read Images
    images = {}
    for i in fileNumList:
        q = cv2.imread(path+"/"+str(i)+".jpg")
        q = cv2.cvtColor( q, cv2.COLOR_RGB2GRAY )
        images[i] = q
    return images

def Label_Data(fileNumList, hori, vert, cropped_img_dim, OVERLAP):  #Classify images with the phone label.
    train_x = []
    conversion = []
    train_y = []
    for img_no in fileNumList:                                      #Go through all the images in the folder                          
        for l in range(int(vert)-1):                                #Cut the image Vertically
            for k in range(int(hori)-1):                            #Cut the image Horizontally
                for i in range(int((cropped_img_dim[0]/OVERLAP)*l),int(cropped_img_dim[0]*((l/OVERLAP)+1))):        #Make new images with fixed height and width (cropped_omg_dim)
                    for j in range(int((cropped_img_dim[1]/OVERLAP)*k),int(cropped_img_dim[1]*((k/OVERLAP)+1))): 
                        conversion.append(images[img_no][i][j])                     
                img = np.reshape(conversion,(cropped_img_dim))                                                      #One patch of an image (Cutted image)
                train_x.append(img)                                                                                 #Build train_x
                conversion = []
                center = (((cropped_img_dim[1]/OVERLAP)*(k+1))/dim[0],((cropped_img_dim[0]/OVERLAP)*(l+1))/dim[1])  #Find_center of the image and normalise it 
                left_t = (((cropped_img_dim[1]/OVERLAP)*k)/dim[0],((cropped_img_dim[0]/OVERLAP)*l)/dim[1])          #Find left top corner of the image and normalise it
                right_b = ((cropped_img_dim[1]*((k/OVERLAP)+1))/dim[0],(cropped_img_dim[0]*((l/OVERLAP)+1))/dim[1]) #Find right bottom of the image and normalise it
                right_t = ((cropped_img_dim[1]*((k/OVERLAP)+1))/dim[0],((cropped_img_dim[0]/OVERLAP)*l)/dim[1])     #Find right top of the image and normalise it
                left_b = (((cropped_img_dim[1]/OVERLAP)*k)/dim[0],(cropped_img_dim[0]*((l/OVERLAP)+1))/dim[1])      #Find right bottom of the image and normalise it
                dist = Find_Distance(phone_coordinates[img_no],center)                                              #Find distance from center and ground truth(Label of phone coordinates)
                if dist <= THRESHOLD:                                                                               
                    train_y.append(np.array(1))
                    for j in range(num_of_duplicates):                                                              #If the ground truth is withing a thresholding range or ....
                        train_x.append(img)                                                                         #the ground truth is inside of the image, then label it 1.
                        train_y.append(np.array(1))                                                                 #Duplicate the phone image                                  
                elif left_t[0] <= phone_coordinates[img_no][0] <=  right_b[0] and left_t[1] <= phone_coordinates[img_no][1] <=  right_b[1]:  
                    train_y.append(np.array(1)) 
                    for j in range(num_of_duplicates):
                            train_x.append(img)
                            train_y.append(np.array(1))
                else:
                    train_y.append(np.array(0))
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y

def Train_Network(train_x,train_y):     #Train Convolution neural network
    nn = Classifier(                    # classification 
        layers=[                                                                                                    
            Convolution('Rectifier', channels=32, kernel_shape=(3,3), kernel_stride = (2,2) ,border_mode='same'), #Conv layer with 3 * 3 filter with 32 neurons
            Convolution('Rectifier', channels=16, kernel_shape=(3,3),kernel_stride = (2,2), border_mode='same'),  #Conv layer with 3 * 3 filter with 16 neurons (Downsampling)
            Convolution('Rectifier', channels=8, kernel_shape=(3,3),kernel_stride = (2,2), border_mode='same'),  #Conv layer with 3 * 3 filter with 8 neurons (Downsampling)
            Layer('Softmax')],          #Softmax 
        learning_rate=0.002,            #Learning Rate                                                                             
        valid_size=0.2,                 #validate 20 % of the images while training
        n_iter = 30,                    #n_iter refers to n_epoch for sknn 
        n_stable=10,                    #stop training after 10 stable iterations. Threshold is set by default as f_stable = 0.001
        batch_size = 50,                #Batch Size 
        normalize = 'Batch',            #Normalization
        verbose=False)                  #if verbose is true, then it shows the interface of the network
    a = nn.fit(train_x, train_y)        #Train the network
    output = open('pickle_file.pkl', 'wb')   #Generate pickle file to store the result of trained network
    pickle.dump(a, output)              #Write the data of the trained network so in testing file it can be fetched.
    output.close()                      #close the pickle file
    #print('\nTRAIN SCORE', nn.score(train_x,train_y))  #It shows Training Score  
read_label_file_path = path+"\labels.txt" 
Label_Exraction(read_label_file_path,write_label_file_path)
phone_coordinates, fileNumList = Read_Given_Labels(write_label_file_path, path)   #Read the file with labels
images = Read_Images(fileNumList, path)                                     #Read images 
image_dim = (len(images[0]),len(images[0][0]))                              #Image dimention
vert = image_dim[0]/(cropped_img_dim[0]/OVERLAP)                            #Number of images make vertically out of one image 
hori = image_dim[1]/(cropped_img_dim[1]/OVERLAP)                            #Number of images make vertically out of one image
train_x,train_y = Label_Data(fileNumList, hori, vert, cropped_img_dim, OVERLAP)         #Label the data (Classify generated images from given images with 1's and 0's)
Train_Network(train_x,train_y)                                              #Train the network
