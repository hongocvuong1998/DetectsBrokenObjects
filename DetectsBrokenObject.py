from Header import *
from Train import *

model = MLP()
serializers.load_npz('DetectsBrokenObject_result/model_epoch-15', model) #load file resule
for root, dirs , files in os.walk('D:\\DetectsBrokenObject\\Dataset\\TestingDataSetScale\\NORMAL\\'):
            for file in files:
                if file.endswith(".jpg"):
                    string=os.path.join(root, file)
                    img=cv2.imread(string)
                    img = cv2.resize(img,(512, 512), interpolation = cv2.INTER_AREA)
                    imggray=rgb2gray(img)
                    imggray=np.asarray(imggray,dtype=np.float32)
                    imggray=np.reshape(imggray,[1,512,512])
                    y = model(imggray[None, ...])
                    result= y.data.argmax(axis=1)[0]
                    strout=''
                    if result==0:
                        strout='DAMAGE'
                    else:
                        strout='NORMAL'
                    print(strout)

# cap = cv2.VideoCapture('D:\\DataStoreData\\Video\\4.mp4')
# font = cv2.FONT_HERSHEY_SIMPLEX
# out = cv2.VideoWriter(VideoOutput+str(4),cv2.VideoWriter_fourcc('M','J','P','G'), 40, (512,512))


# while(cap.isOpened()):
#     ret, frame = cap.read()
#     #print('type ret : ',type(ret))
#     frame=crop_center(frame,240,240)
#     framegray=rgb2gray(frame)

#     framegray=np.asarray(framegray,dtype=np.float32)
#     framegray=np.reshape(framegray,[1,240,240])
#     # print('shape image: ', framegray.shape)
#     # exit()
    
#     #print('dim frame', framegray.ndim)
#     #exit()
#     #print('type frame : ',type(frame))
#     y = model(framegray[None, ...])
#     result= y.data.argmax(axis=1)[0]
#     strout=''
#     if result==0:
#         strout='DAMAGE'
#     else:
#         strout='NORMAL'
   
#     cv2.putText(frame,strout,(10,10), font, 1,(0,0,255),2) #Draw the text
#     #print('img shape', frame.shape)
#     #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     out.write(frame)
#     cv2.imshow('frame',frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
