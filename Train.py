from PrepareDataset import *
from Model import *
from Header import *

train = PreprocessedDataset(AddressDatasetCSV[0], Root)
val = PreprocessedDataset(AddressDatasetCSV[1], Root)
test=PreprocessedDataset(AddressDatasetCSV[2], Root)

gpu_id = -1 # Set to -1 if you use CPU

batchsize=16

train_iter=iterators.SerialIterator(train,batchsize)
test_iter=iterators.SerialIterator(val,batchsize,False,False)

#train_iter = chainer.iterators.MultiprocessIterator(train, batchsize)
#test_iter = chainer.iterators.MultiprocessIterator(val, batchsize, repeat=False,shuffle=False )

#model = resnet50.ResNeXt50() 
#model=Alexnet() ## Don't working on Gray image

model=MLP()

if gpu_id >= 0:
    model.to_cpu(gpu_id)
max_epoch=500
model=L.Classifier(model)

optimizer = chainer.optimizers.MomentumSGD(lr=0.00001, momentum=0.9)#

#optimizer = chainer.optimizers.Adam(alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-08)

optimizer.setup(model)

updater=training.updaters.StandardUpdater(train_iter,optimizer,device=gpu_id)
trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='DetectsBrokenObject_result') #save result model
trainer.extend(extensions.LogReport()) #save loss and accuracy
trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.snapshot_object(model.predictor, filename='model_epoch-{.updater.epoch}')) #save snapshot epoch
trainer.extend(extensions.Evaluator(test_iter, model, device=gpu_id)) 
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy','validation/main/loss', 'validation/main/accuracy', 'elapsed_time'])) # Print in the screen
trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png')) # Draw loss.png and save in dir
trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png')) # Draw loss.png and save in dir
trainer.extend(extensions.dump_graph('main/loss'))
chainer.serializers.load_npz('D:\\DetectsBrokenObject\\DetectsBrokenObject_result\\snapshot_epoch-241', trainer)
print('123')
trainer.run()


