import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from PIL import Image
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer.datasets import mnist
import chainer.datasets as datasets
from functools import partial
import time
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import resnet50
from chainer import initializers
k=1
AddressDatasetCSV=['E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSet.csv',
                   'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSet.csv',
                   'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSet.csv']
Root='D:\\DetectsBrokenObject\\Dataset\\'
VideoOutput='D:\\DetectsBrokenObject\\Videooutput\\'
AddressImgBeforeScale = [
    'D:\\DetectsBrokenObject\\DataStoreData\\TrainingDataSetScale\\DAMAGE', #Target 0
    'D:\\DetectsBrokenObject\\DataStoreData\\TrainingDataSetScale\\NORMAL', #Target 1
    
    'D:\DetectsBrokenObject\\DataStoreData\\EvaluationDataSetScale\\DAMAGE', #Target 0 
    'D:\DetectsBrokenObject\\DataStoreData\\EvaluationDataSetScale\\NORMAL', #Target 1

    'D:\\DetectsBrokenObject\\DataStoreData\\TestingDataSetScale\\DAMAGE', #Target 0
    'D:\\DetectsBrokenObject\\DataStoreData\\TestingDataSetScale\\NORMAL'  #Target 1
]

AddressDataset=[
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSetScale\\DAMAGE',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSetScale\\NORMAL',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSetScale\\DAMAGE',
    'E:\\MyData\\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSetScale\\NORMAL',
    'E:\\MyData\\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSetScale\\DAMAGE',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSetScale\\NORMAL'
]
TrainImgList=[]
TargetTrainImgList=[]
ValImgList=[]
TargetValImgList=[]
TestImgList=[]
TargetTestImgList=[]

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
def crop_center(img,cropx,cropy):
    y,x,z = img.shape  #z: RGB z=3
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)   
    return img[starty:starty+cropy,startx:startx+cropx]

def SaveImgAfterCrop(img,Address):
    folderSaveImg='D:\\DetectsBrokenObject\\Dataset\\' 
    img=crop_center(img,512,512)
    if 'TrainingDataSetScale\\DAMAGE' in Address :
        folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\DAMAGE\\'
    elif 'TrainingDataSetScale\\NORMAL' in Address:
        folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\NORMAL\\'
    elif 'EvaluationDataSetScale\\DAMAGE' in Address:
        folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\DAMAGE\\'
    elif 'EvaluationDataSetScale\\NORMAL' in Address:
        folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\NORMAL\\'
    elif 'TestingDataSetScale\\DAMAGE' in Address:
        folderSaveImg=folderSaveImg + 'TestingDataSetScale\\DAMAGE\\'
    else: # TestingDataSetScale\\NORMAL
        folderSaveImg=folderSaveImg + 'TestingDataSetScale\\NORMAL\\'

    namefile='.jpg'  # Define file imgage output
    global k 
    k=k+1
    addr=  folderSaveImg + str(k) + namefile
    img = Image.fromarray(img)
    img.save(addr)
def CreatFileCSV(pathImg):
    global k
    path=pathImg
    print(path)
    path=path.replace('D:\\DetectsBrokenObject\\Dataset\\', '', 1)
    if 'DAMAGE' in path:
        path = path + ' 0'
    else:
        path=path + ' 1'
    
    print(k,'   ',path)
    k=k+1
    
    EvaluationDataSet=open('E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSet.csv','a')
    TestingDataSet=open('E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSet.csv','a')
    TrainingDataSet=open('E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSet.csv', 'a')

    
    if 'EvaluationDataSet' in path:
        EvaluationDataSet.write(path)
        EvaluationDataSet.write('\n')
        EvaluationDataSet.close()
    elif 'TrainingDataSet' in path:
        TrainingDataSet.write(path)
        TrainingDataSet.write('\n')
        TrainingDataSet.close()
    else:
        TestingDataSet.write(path)
        TestingDataSet.write('\n')
        TestingDataSet.close()
def ReadImage(): # Đọc file vào
    k=1
    global TargetTrainImgList
    global TargetTestImgList
    global TargetValImgList

    for i in range(6): # i = 0 1 2 3 
        for root, dirs , files in os.walk(AddressDataset[i]):
            for file in files:
                if file.endswith(".jpg"):
                    string=os.path.join(root, file)
                    CreatFileCSV(string)
                    '''
                    img=mpimg.imread(string)
                    print(k,'   img shape', img.shape)
                    k=k+1
                    img=rgb2gray(img)
                    img *=(1.0 / 255.0)
                    #print(img)
                                   

                    #SaveImgAfterCrop(img,AddressImgBeforeScale[i]) #If not done crop

                    img=np.asarray(img,dtype=np.float32)
                    #print('type(img):',img[0].dtype)
                    img=np.reshape(img,[1,512,512])

                    

                    if AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\DAMAGE':
                        TrainImgList.append(img)
                        TargetTrainImgList.append(0)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\NORMAL':
                        TrainImgList.append(img)
                        TargetTrainImgList.append(1)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\DAMAGE':
                        ValImgList.append(img)
                        TargetValImgList.append(0)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\NORMAL':
                        ValImgList.append(img)
                        TargetValImgList.append(1)
                    elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\DAMAGE':
                        TestImgList.append(img)
                        TargetTestImgList.append(0)
                    else:
                        TestImgList.append(img)
                        TargetTestImgList.append(1)
                    # # no=gc.collect()
                    # print('no gc collect:',no)


    TargetTrainImgList = np.asarray(TargetTrainImgList,dtype=np.int32)
    TargetTestImgList = np.asarray(TargetTestImgList,dtype=np.int32)
    TargetValImgList = np.asarray(TargetValImgList,dtype=np.int32)\

    Mytrain=datasets.TupleDataset(TrainImgList, TargetTrainImgList)
    Myval=datasets.TupleDataset(ValImgList, TargetValImgList)
    Mytest=datasets.TupleDataset(TestImgList, TargetTestImgList)
    return Mytrain,Myval,Mytest
    '''
class PreprocessedDataset(chainer.dataset.DatasetMixin):

    def __init__(self, path, root):
        self.base = chainer.datasets.LabeledImageDataset(path, root)

    def __len__(self):
        return len(self.base)

    def get_example(self, i):
       
        # It reads the i-th image/label pair and return a preprocessed image.
        image, label = self.base[i]
        #image=np.rollaxis(image, 0, 3)
        #image=crop_center(image,224,224)

        #image=rgb2gray(image)
        
        image *= (1.0 / 255.0) # Scale to [0, 1]
        
        image=np.asarray(image,dtype=np.float32)
        
        #image=np.reshape(image,[1,512,512])
        # print(image)
        # print('shape image: ', image.shape)
        # print('type(img):',image[0].dtype)
        #exit()
        return image, label
