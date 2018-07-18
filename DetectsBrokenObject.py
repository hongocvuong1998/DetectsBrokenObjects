from Header import *
from Train import *

# ReadImage()
# exit()       
#################Insert read dataset ################

# class PreprocessedDataset(chainer.dataset.DatasetMixin):

#     def __init__(self, path, root):
#         self.base = chainer.datasets.LabeledImageDataset(path, root)

#     def __len__(self):
#         return len(self.base)

#     def get_example(self, i):
       
#         # It reads the i-th image/label pair and return a preprocessed image.
#         image, label = self.base[i]
#         image=np.rollaxis(image, 0, 3)
#         #image=crop_center(image,224,224)
#         image=rgb2gray(image)
#         image *= (1.0 / 255.0) # Scale to [0, 1]
        
#         image=np.asarray(image,dtype=np.float32)
#         #print('type(img):',img[0].dtype)
#         image=np.reshape(image,[1,512,512])
        
#         return image, label

# train = PreprocessedDataset(AddressDatasetCSV[0], Root)
# val = PreprocessedDataset(AddressDatasetCSV[1], Root)
# test=PreprocessedDataset(AddressDatasetCSV[2], Root)

#######################################################  

#train,test=ReadImage()



###############Insert your model in here#################


# from functools import partial
# class LeNet5(Chain):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         net = [('conv1', L.Convolution2D(1, 6, 5, 1))]
#         net += [('_sigm1', F.sigmoid)]
#         net += [('_mpool1', partial(F.max_pooling_2d, ksize=2, stride=2))]
#         net += [('conv2', L.Convolution2D(6, 16, 5, 1))]
#         net += [('_sigm2', F.sigmoid)]
#         net += [('_mpool2', partial(F.max_pooling_2d, ksize=2, stride=2))]
#         net += [('conv3', L.Convolution2D(16, 120, 4, 1))]
#         net += [('_sigm3', F.sigmoid)]
#         net += [('_mpool3', partial(F.max_pooling_2d, ksize=2, stride=2))]
#         net += [('fc4', L.Linear(None, 84))]
#         net += [('_sigm4', F.sigmoid)]
#         net += [('fc5', L.Linear(84, 2))]
#         net += [('_sigm5', F.sigmoid)]
#         with self.init_scope():
#             for n in net:
#                 if not n[0].startswith('_'):
#                     setattr(self, n[0], n[1])
#         self.forward = net

#     def __call__(self, x):
#         for n, f in self.forward:
#             # print(n)
#             if not n.startswith('_'):
#                 x = getattr(self, n)(x)
#             else:
#                 x = f(x)
#         if chainer.config.train:
#             return x
#         return F.softmax(x)


# class MLP(Chain):     #first Model
#     def __init__(self, n_mid_units=100, n_out=2):
#         super(MLP, self).__init__()
#         with self.init_scope():
#             self.l1 = L.Linear(None, n_mid_units)
#             self.l2 = L.Linear(None, n_mid_units)
#             self.l3 = L.Linear(None, n_out)
#     def __call__(self, x):
#         h1 = F.relu(self.l1(x))
#         h2 = F.relu(self.l2(h1))
#         return self.l3(h2)

# #######################################################

# gpu_id = -1 # Set to -1 if you use CPU

# batchsize=5

# train_iter=iterators.SerialIterator(train,batchsize)
# test_iter=iterators.SerialIterator(val,batchsize,False,False)

# '''
# train_iter = chainer.iterators.MultiprocessIterator(train, batchsize, n_processes=args.loaderjob)
# val_iter = chainer.iterators.MultiprocessIterator(val,batchsize, repeat=False, n_processes=args.loaderjob)
# '''
# if gpu_id >= 0:
#     MLP().to_cpu(gpu_id)
# max_epoch=100
# model=L.Classifier(MLP())


# optimizer=optimizers.MomentumSGD(lr=0.0001)#


# optimizer.setup(model)

# updater=training.updaters.StandardUpdater(train_iter,optimizer,device=gpu_id)
# trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='DetectsBrokenObject_result') #save result model
# trainer.extend(extensions.LogReport()) #save loss and accuracy
# trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
# trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}')) #save snapshot epoch
# trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id)) 
# trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy', 'elapsed_time'])) # Print in the screen
# trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png')) # Draw loss.png and save in dir
# trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png')) # Draw loss.png and save in dir
# trainer.extend(extensions.dump_graph('main/loss'))
#trainer.run()


model = MLP()
serializers.load_npz('DetectsBrokenObject_result/model_epoch-7', model) #load file resule

cap = cv2.VideoCapture('D:\\DataStoreData\\Video\\4.mp4')
font = cv2.FONT_HERSHEY_SIMPLEX
out = cv2.VideoWriter(VideoOutput+str(4),cv2.VideoWriter_fourcc('M','J','P','G'), 40, (512,512))


while(cap.isOpened()):
    ret, frame = cap.read()
    #print('type ret : ',type(ret))
    frame=crop_center(frame,512,512)
    framegray=rgb2gray(frame)

    framegray=np.asarray(framegray,dtype=np.float32)
    framegray=np.reshape(framegray,[1,512,512])
    #print('dim frame', framegray.ndim)
    #exit()
    #print('type frame : ',type(frame))
    y = model(framegray[None, ...])
    result= y.data.argmax(axis=1)[0]
    strout=''
    if result==0:
        strout='DAMAGE'
    else:
        strout='NORMAL'
   
    cv2.putText(frame,strout,(10,500), font, 1,(0,0,255),2) #Draw the text
    #print('img shape', frame.shape)
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
'''

for i in range(len(test)):
    x, t = test[i]
    #out.write(x)
    #cv2.imshow('img',x)
    
    # cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #time.sleep(1)
    #print('shape : ', x.shape)
    
    #plt.imshow(x.reshape(512, 512), cmap='gray')
    #plt.show()

    print('label:', t)
    y = model(x[None, ...])
    result=y.data.argmax(axis=1)[0]
    print('predicted_label:',result)
'''