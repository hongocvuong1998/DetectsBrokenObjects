import time

import numpy as np
import cv2

'''
np.random.seed(0)
#a = np.array([[1,2],[1,2]])
#a = np.reshape(a,[1,2,2])

b = np.array([[1,2],[1,2]])
a = np.array([])
print(b)
print ('shape a ', a.shape)
#np.insert(a, 1, 5)
np.append(a,)
print(a)
print ('shape a ', a.shape)
'''
'''
import numpy as np
a = np.array([[1, 1], [2, 2], [3, 3]])
# print('dim a', a.dim)
b = np.array([[4, 4], [2, 2], [3, 3]])
c = np.array([[4, 4], [2, 2], [3, 3]])
print ('shape a ', a.shape)
x=[a]
x.append(b)
x.append(c)
x = np.array(x)
print ('shape x ', x.shape)





x = []
print("Original array:")
print(x)
x = np.append(x, [[40, 50, 60], [70, 80, 90]])
print ('shape x ', x.shape)
print("After append values to the end of the array:")
print(x)
'''
#description='Process some integers.'  integers
# import csv
 
# AData = ['first_name', 'second_name', 'Grade']
# BData = ['Alex', 'Brian', 'A'],
# CData =  ['Tom', 'Smith', 'B']
 
# myFile = open('example1.csv', 'w')
# '''
#     writer = csv.writer(myFile)
#     writer.writerows(AData)
#     writer = csv.writer(myFile)
#     writer.writerows(BData)
#     writer = csv.writer(myFile)
#     writer.writerows(CData)
# '''
# myFile.write('Hello 2018')     
# print("Writing complete")

#print(args.sum)

'''
l=[1,2,3,4]p=[-2]
print(p)
'''
# import numpy as np
# import cv2
#import sys
#sys.path.append(r"C:\Users\VuongHN\AppData\Local\Programs\Python\Python36-32\Lib\site-packages")

# import numpy as np


# cap = cv2.VideoCapture('D:\\DataStoreData\\Video\\1.mp4')
# font = cv2.FONT_HERSHEY_SIMPLEX
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 300, (512,512))
# while(cap.isOpened()):
#     ret, frame = cap.read()
#     #print('type ret : ',type(ret))
#     frame=crop_center(frame,512,512)
#     #print('type frame : ',type(frame))
#     cv2.putText(frame,'OpenCV',(10,500), font, 1,(0,0,255),2) #Draw the text
#     #print('img shape', frame.shape)
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

# from Header import *

# from chainer.links.caffe import CaffeFunction

# AddressDataset=[
#     'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\DAMAGE',
#     'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\NORMAL',
#     'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\DAMAGE',
#     'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\NORMAL',
#     'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\DAMAGE',
#     'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\NORMAL'
# ]
folderSaveImg=['D:\\Myfolder\\Temp\\Train\\DAMAGE\\',
'D:\\Myfolder\\Temp\\Train\\NORMAL\\',
'D:\\Myfolder\\Temp\\Eva\\DAMAGE\\',
'D:\\Myfolder\\Temp\\Eva\\NORMAL\\',
'D:\\Myfolder\\Temp\\Test\\DAMAGE\\',
'D:\\Myfolder\\Temp\\Test\\NORMAL\\']

'''
]
i=4
for i in range(6): # i = 0 1 2 3 
    for root, dirs , files in os.walk(AddressDataset[i]):
        for file in files:
            if file.endswith(".jpg"):
                k=k+1
                string=os.path.join(root, file)
                #print(string)
                CreatFileCSV(string)
                namefile='.jpg'
                image=cv2.imread(string)
                #image = cv2.resize(image,None,fx=0.8, fy=0.8, interpolation = cv2.INTER_LINEAR)
                image=crop_center(image, 512, 512)
                #image = cv2.resize(image,(512, 512), interpolation = cv2.INTER_AREA)
                #addr=  folderSaveImg[i] + str(k)+'.0' + namefile
                print(string)
                cv2.imwrite(string,image)

                # w,h,z=image.shape
                # center = (w / 2, h / 2)
                # M = cv2.getRotationMatrix2D(center, 90, 1.0)
                # rotated = cv2.warpAffine(image, M, (w, h))
                
                # addr=  folderSaveImg[i] + str(k)+'.90' + namefile
                # cv2.imwrite(addr,rotated)

                # M = cv2.getRotationMatrix2D(center, 180, 1.0)
                # rotated = cv2.warpAffine(image, M, (w, h))
                
                # addr=  folderSaveImg[i] + str(k)+'.180' + namefile
                # cv2.imwrite(addr,rotated)

                # M = cv2.getRotationMatrix2D(center, 270, 1.0)
                # rotated = cv2.warpAffine(image, M, (w, h))
                
                # addr=  folderSaveImg[i] + str(k)+'.270' + namefile
                # cv2.imwrite(addr,rotated)
                    #print(img)
                                   

                    #SaveImgAfterCrop(img,AddressImgBeforeScale[i]) #If not done crop

                    #img=np.asarray(img,dtype=np.float32)
                    #print('type(img):',img[0].dtype)
                   # img=np.reshape(img,[1,512,512])

                    


image=cv2.imread("1.dcm")
#image=crop_center(image, 480, 480)
#print(' image shape', image.shape)
cv2.imshow("Image", image)
#w,h,z=image.shape
cv2.imshow('img',image)
#center = (w / 2, h / 2)
#M = cv2.getRotationMatrix2D(center, 360, 1.0)
#rotated = cv2.warpAffine(image, M, (w, h))

cv2.waitKey(0)
#D:\Myfolder\DatasetForDeepLearning\DataProcess\EvaluationDataSetScale\DAMAGE
'''
#______________________________ RENAME FILE______________________________________#
# i = 1
# folder="D:\\DetectsBrokenObject\\DataStoreData\\DatabyTamNT27\\"
# for filename in os.listdir(folder):
#     dst =str(i) + ".jpg"
#     src =folder+ filename
#     #print(src)
#     dst =folder+ dst
#     #print('========>',dst)        
#     # rename() function will
#     # rename all the files
#     os.rename(src, dst)
#     i += 1


#________________________________________________________ BOUNDING BOX OBJECT ________________________________________________#
# import cv2
# import numpy as np

# import random as rng
# import math
# rng.seed(12345)





# def Distance(Point1,Point2):
#     X1,Y1=Point1
#     X2,Y2=Point2
#     dx=abs(X2-X1)
#     dy=abs(Y2-Y1)
#     return math.sqrt(math.pow(dx,2)+math.pow(dy,2))

# def nothing(x):
#   pass
# def thresh_callback(val):
#     threshold = val
    
#     canny_output = cv2.Canny(framegray, threshold, threshold * 2)
    
    
#     _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
#     contours_poly = [None]*len(contours)
#     boundRect = [None]*len(contours)
#     centers = [None]*len(contours)
#     radius = [None]*len(contours)




#     for i in range(len(contours)):
#         contours_poly[i] = cv2.approxPolyDP(contours[i], 3, True)
#         boundRect[i] = cv2.boundingRect(contours_poly[i])
#         centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])
    
    
#     drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
    
#     x_min=512
#     y_min=512
#     x_max=0
#     y_max=0  
    
#     for i in range(len(contours)):
#         color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
#         cv2.drawContours(drawing, contours_poly, i, color)
#         x1=int(boundRect[i][0])
#         y1=int(boundRect[i][1])
#         x2=int(boundRect[i][0]+boundRect[i][2])
#         y2=int(boundRect[i][1]+boundRect[i][3])
        
#         #print('( ',x1,'  ',y1, ')   and  ( ' ,x2,'  ',y2, ')')
#         if (x2-x1)*(y2-y1)<1000:
            
#             continue
#         if x1 <x_min:
#             x_min=x1
#         if y1<y_min : 
#             y_min=y1 
#         if x2>x_max :
#             x_max=x2 
#         if y2>y_max:
#             y_max=y2 
        
#         #cv2.rectangle(drawing, (int(boundRect[i][0]), int(boundRect[i][1])), (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
#         #cv2.circle(drawing, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), color, 2)

#     Point1=(x_min,y_min)
#     Point2=(x_max,y_max)
#     cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max), color, 2)
#     cv2.rectangle(drawing, Point1, Point2, color, 2)
#     #print('( ',x_min,'  ',y_min, ')   and  ( ' ,x_max,'  ',y_max, ')')
#     #cv2.imshow('Contours', drawing)
#     return Point1, Point2
    
# cap = cv2.VideoCapture('D:\\DetectsBrokenObject\\DataStoreData\\Video\\4.mp4')
# font = cv2.FONT_HERSHEY_SIMPLEX
# frame_width = int(cap.get(3))
# frame_height = int(cap.get(4))
# out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width,frame_height))



# max_thresh = 255
# thresh = 100 # initial threshold
# prePoint1=(0,0)
# prePoint2=(0,0)
# preDistance = 0

# source_window = "Source"
# #cv2.namedWindow(source_window)
# #cv2.createTrackbar("Canny thresh:", source_window, thresh, max_thresh, nothing)

# ___________________________________CROP IMAGE_____________________________________________#
# lower_blue=np.array([80,130,10])
# upper_blue=np.array([130,255,255])

# folderR='D:\\DetectsBrokenObject\\DataStoreData\\DatabyTamNT27\\'

# folderS='D:\\DetectsBrokenObject\\DataStoreData\\BoundingBoxDataTamNT27\\'
# k=0

# # for i in range(6): # i = 0 1 2 3 
# for root, dirs , files in os.walk(folderR):
#     for file in files:
#         if file.endswith(".jpg"):
#             k=k+1
#             string=os.path.join(root, file)
#             print(string)
            
#             namefile='.jpg'
#             frame=cv2.imread(string)


#             #frame=crop_center(frame,972,1728)
#             frame = cv2.resize(frame,None,fx=0.2, fy=0.2, interpolation = cv2.INTER_LINEAR)

            


#             # framegray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            
#             # ret,framegray = cv2.threshold(framegray,98,255,cv2.THRESH_BINARY)
#             # kernel = np.ones((5,5),np.uint8)
#             # framegray = cv2.erode(framegray,kernel,iterations = 1)
            
#             mask = cv2.inRange(hsv, lower_blue, upper_blue)
#             framegray = cv2.bitwise_not(mask)
#             #cv2.imshow('mask',framegray)
            
            
#             Point1,Point2 = thresh_callback(180)
            
#             x1,y1=Point1
#             x2,y2=Point2
#             cv2.rectangle(frame, Point1, Point2, (0,0,255), 2)
#             #cv2.imshow('frame', frame)
#             #cv2.waitKey(0)
#             img= frame[y1:y2,x1:x2]
#             #img=cv2.resize(img,(240, 240), interpolation = cv2.INTER_AREA)
#             # cv2.imshow('Bound', frame)
#             #cv2.imshow('img', img)
#             #time.sleep(.1)
#             #cv2.waitKey(1)
#             savefile=folderS+str(k)+namefile
#             #print(savefile)
#             cv2.imwrite(savefile, frame)
# print(k)


#____________________________READ IMAGE FROM VIDEO_________________________________#


# while(cap.isOpened()):
#     #ret, frame = cap.read()
#     frame=cv2.imread("D:\\DetectsBrokenObject\\DataStoreData\\DataByVuong\\Image\\1\\DAMAGE\\72.jpg")
#     frame = cv2.resize(frame,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

   
#     framegray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     framegray = cv2.blur(framegray, (3,3))
#     # cv2.imshow('frame gray',framegray)
        
#     thresh=cv2.getTrackbarPos("Canny thresh:" , source_window)
#     print(thresh)
#     Point1,Point2 = thresh_callback(thresh)

#     if Distance(Point1,Point2)>1000:
#         Point1=prePoint1
#         Point2=prePoint2
#     prePoint1=Point1
#     prePoint2=Point2
#     # print(Distance(Point1,Point2))
#     x1,y1=Point1
    
#     cv2.rectangle(frame, Point1, Point2, (0,0,255), 2)
#     cv2.putText(frame,'Detects Broken Object',(x1, y1-10), font, 1,(0,0,255),2) #Draw the text
#     cv2.imshow(source_window, frame)

#     #out.write(frame)
#     #cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
def takeSecond(elem):
    return elem[0]*elem[1]
        

# a=[[1,2],[10,20],[5,10],[6,12],[3,6]]
# a.sort(key=takeSecond)
a=(1,2,3,4)
x1,y1,x2,y2=a
print(x1,y1,x2,y2)
print('type ', type(a))













#FROM HEADER

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
# def crop_center(img,cropx,cropy):
#     y,x,z = img.shape  #z: RGB z=3
#     startx = x//2-(cropx//2)
#     starty = y//2-(cropy//2)   
#     return img[starty:starty+cropy,startx:startx+cropx]

# def SaveImgAfterCrop(img,Address):
#     folderSaveImg='D:\\DetectsBrokenObject\\Dataset\\' 
#     img=crop_center(img,512,512)
#     if 'TrainingDataSetScale\\DAMAGE' in Address :
#         folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\DAMAGE\\'
#     elif 'TrainingDataSetScale\\NORMAL' in Address:
#         folderSaveImg=folderSaveImg + 'TrainingDataSetScale\\NORMAL\\'
#     elif 'EvaluationDataSetScale\\DAMAGE' in Address:
#         folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\DAMAGE\\'
#     elif 'EvaluationDataSetScale\\NORMAL' in Address:
#         folderSaveImg=folderSaveImg + 'EvaluationDataSetScale\\NORMAL\\'
#     elif 'TestingDataSetScale\\DAMAGE' in Address:
#         folderSaveImg=folderSaveImg + 'TestingDataSetScale\\DAMAGE\\'
#     else: # TestingDataSetScale\\NORMAL
#         folderSaveImg=folderSaveImg + 'TestingDataSetScale\\NORMAL\\'

#     namefile='.jpg'  # Define file imgage output
#     global k 
#     k=k+1
#     addr=  folderSaveImg + str(k) + namefile
#     img = Image.fromarray(img)
#     img.save(addr)
# def CreatFileCSV(pathImg):
#     global k
#     path=pathImg
#     print(path)
#     path=path.replace('D:\\DetectsBrokenObject\\Dataset\\', '', 1)
#     if 'DAMAGE' in path:
#         path = path + ' 0'
#     else:
#         path=path + ' 1'
    
#     print(k,'   ',path)
#     k=k+1
    
#     EvaluationDataSet=open('D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSet.csv','a')
#     TestingDataSet=open('D:\\DetectsBrokenObject\\Dataset\\TestingDataSet.csv','a')
#     TrainingDataSet=open('D:\\DetectsBrokenObject\\Dataset\\TrainingDataSet.csv', 'a')

    
#     if 'EvaluationDataSet' in path:
#         EvaluationDataSet.write(path)
#         EvaluationDataSet.write('\n')
#         EvaluationDataSet.close()
#     elif 'TrainingDataSet' in path:
#         TrainingDataSet.write(path)
#         TrainingDataSet.write('\n')
#         TrainingDataSet.close()
#     else:
#         TestingDataSet.write(path)
#         TestingDataSet.write('\n')
#         TestingDataSet.close()
# def ReadImage(): 
#     k=1
#     global TargetTrainImgList
#     global TargetTestImgList
#     global TargetValImgList

#     for i in range(6): # i = 0 1 2 3 
#         for root, dirs , files in os.walk(AddressDataset[i]):
#             for file in files:
#                 if file.endswith(".jpg"):
#                     string=os.path.join(root, file)
#                     CreatFileCSV(string)
                    
#     #                 img=mpimg.imread(string)
#     #                 print(k,'   img shape', img.shape)
#     #                 k=k+1
#     #                 img=rgb2gray(img)
#     #                 img *=(1.0 / 255.0)
#     #                 #SaveImgAfterCrop(img,AddressImgBeforeScale[i]) #If not done crop

#     #                 img=np.asarray(img,dtype=np.float32)
#     #                 #print('type(img):',img[0].dtype)
#     #                 img=np.reshape(img,[1,512,512])

                    

#     #                 if AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\DAMAGE':
#     #                     TrainImgList.append(img)
#     #                     TargetTrainImgList.append(0)
#     #                 elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TrainingDataSetScale\\NORMAL':
#     #                     TrainImgList.append(img)
#     #                     TargetTrainImgList.append(1)
#     #                 elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\DAMAGE':
#     #                     ValImgList.append(img)
#     #                     TargetValImgList.append(0)
#     #                 elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\EvaluationDataSetScale\\NORMAL':
#     #                     ValImgList.append(img)
#     #                     TargetValImgList.append(1)
#     #                 elif AddressDataset[i] == 'D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\DAMAGE':
#     #                     TestImgList.append(img)
#     #                     TargetTestImgList.append(0)
#     #                 else:
#     #                     TestImgList.append(img)
#     #                     TargetTestImgList.append(1)
#     #                 # # no=gc.collect()
#     #                 # print('no gc collect:',no)


#     # TargetTrainImgList = np.asarray(TargetTrainImgList,dtype=np.int32)
#     # TargetTestImgList = np.asarray(TargetTestImgList,dtype=np.int32)
#     # TargetValImgList = np.asarray(TargetValImgList,dtype=np.int32)\

#     # Mytrain=datasets.TupleDataset(TrainImgList, TargetTrainImgList)
#     # Myval=datasets.TupleDataset(ValImgList, TargetValImgList)
#     # Mytest=datasets.TupleDataset(TestImgList, TargetTestImgList)
#     # return Mytrain,Myval,Mytest


# class PreprocessedDataset(chainer.dataset.DatasetMixin):

#     def __init__(self, path, root):
#         self.base = chainer.datasets.LabeledImageDataset(path, root)

#     def __len__(self):
#         return len(self.base)

#     def get_example(self, i):
       
#         # It reads the i-th image/label pair and return a preprocessed image.
#         image, label = self.base[i]
#         image=np.rollaxis(image, 0, 3)
#         #image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_LINEAR)
#         image = cv2.resize(image,(512, 512), interpolation = cv2.INTER_AREA)
#         #image=crop_center(image,224,224)

#         image=rgb2gray(image)
        
#         w,h=image.shape
#         # print('w:  ', w , '    h: ', h)
#         # exit()
#         image *= (1.0 / 255.0) # Scale to [0, 1]
        
#         image=np.asarray(image,dtype=np.float32)
        
#         image=np.reshape(image,[1,w,h])

#         # print('shape image: ', image.shape)
#         # exit()
#         # print('type(img):',image[0].dtype)
#         #exit()
#         return image, label
