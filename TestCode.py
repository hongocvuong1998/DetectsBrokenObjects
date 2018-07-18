'''
import numpy as np
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

from Header import *
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
#from Header import *

# from chainer.links.caffe import CaffeFunction

AddressDataset=[
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSetScale\\DAMAGE',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TrainingDataSetScale\\NORMAL',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSetScale\\DAMAGE',
    'E:\\MyData\\Project\\DetectsBrokenObjects\\Dataset\\EvaluationDataSetScale\\NORMAL',
    'E:\\MyData\\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSetScale\\DAMAGE',
    'E:\\MyData\Project\\DetectsBrokenObjects\\Dataset\\TestingDataSetScale\\NORMAL'
]
# folderSaveImg=['D:\\My folder\\Temp\\Train\\DAMAGE\\',
# 'D:\\My folder\\Temp\\Train\\NORMAL\\',
# 'D:\\My folder\\Temp\\Eva\\DAMAGE\\',
# 'D:\\My folder\\Temp\\Eva\\NORMAL\\',
# 'D:\\My folder\\Temp\\Test\\DAMAGE\\',
# 'D:\\My folder\\Temp\\Test\\NORMAL\\',


# i=4
# for i in range(6): # i = 0 1 2 3 
#     for root, dirs , files in os.walk(AddressDataset[i]):
#         for file in files:
#             if file.endswith(".jpg"):
#                 k=k+1
#                 string=os.path.join(root, file)
#                 #print(string)
#                 CreatFileCSV(string)
#                 namefile='.jpg'
#                 image=cv2.imread(string)
#                 #image = cv2.resize(image,None,fx=0.8, fy=0.8, interpolation = cv2.INTER_LINEAR)
#                 image=crop_center(image, 512, 512)
#                 #image = cv2.resize(image,(512, 512), interpolation = cv2.INTER_AREA)
#                 #addr=  folderSaveImg[i] + str(k)+'.0' + namefile
# /

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

                    


# image=cv2.imread("Test.jpg")
# image=crop_center(image, 480, 480)
# print('0 img', image.shape)
# w,h,z=image.shape
# cv2.imshow('img',image)
# center = (w / 2, h / 2)
# M = cv2.getRotationMatrix2D(center, 360, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("rotated", rotated)
# cv2.waitKey(0)
#D:\Myfolder\DatasetForDeepLearning\DataProcess\EvaluationDataSetScale\DAMAGE

############################################ RENAME FILE

i = 1247
folder1="E:\\MyData\\Image\\Object\\Image\\12\\DAMAGE\\"
folder2="E:\\MyData\\Image\\Object\\Image\\12\\NORMAL\\"
for filename in os.listdir(folder1):
    dst =str(i) + ".jpg"
    src =folder1+ filename
    dst =folder1+ dst

        
    # rename() function will
    # rename all the files
    os.rename(src, dst)
    i += 1
for filename in os.listdir(folder2):
    dst =str(i) + ".jpg"
    src =folder2+ filename
    dst =folder2+ dst

        
    # rename() function will
    # rename all the files
    os.rename(src, dst)
    i += 1
print(i)
################################################################################################


# loadpath = "bvlc_alexnet.caffemodel"
# savepath = "bvlc_alexnet.chainermodel.pkl"

# from chainer.links.caffe import CaffeFunction
# alexnet = CaffeFunction(loadpath)

# import pickle
# pickle.dump(alexnet, open(savepath, 'wb'))