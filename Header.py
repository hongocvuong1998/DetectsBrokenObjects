import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import math
import random as rng
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
AddressDatasetCSV=['D:\\DetectsBrokenObject\\Dataset\\TrainingDataSet.csv',
                   'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSet.csv',
                   'D:\\DetectsBrokenObject\\Dataset\\TestingDataSet.csv']
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
    'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\DAMAGE',
    'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\NORMAL',
    'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\DAMAGE',
    'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\NORMAL',
    'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\DAMAGE',
    'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\NORMAL'
]
TrainImgList=[]
TargetTrainImgList=[]
ValImgList=[]
TargetValImgList=[]
TestImgList=[]
TargetTestImgList=[]
ListRect=[]
