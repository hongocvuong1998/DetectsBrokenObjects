from Header import *

train = PreprocessedDataset(AddressDatasetCSV[0], Root)
val = PreprocessedDataset(AddressDatasetCSV[1], Root)
test=PreprocessedDataset(AddressDatasetCSV[2], Root)
class MLP(Chain):     #first Model
    def __init__(self, n_mid_units=100, n_out=2):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_out)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

#from functools import partial


class LeNet5(Chain):
    def __init__(self):
        super(LeNet5, self).__init__()
        net = [('conv1', L.Convolution2D(1, 6, 5, 1))]
        net += [('_sigm1', F.sigmoid)]
        net += [('_mpool1', partial(F.max_pooling_2d, ksize=2, stride=2))]
        net += [('conv2', L.Convolution2D(6, 16, 5, 1))]
        net += [('_sigm2', F.sigmoid)]
        net += [('_mpool2', partial(F.max_pooling_2d, ksize=2, stride=2))]
        net += [('conv3', L.Convolution2D(16, 120, 4, 1))]
        net += [('_sigm3', F.sigmoid)]
        net += [('_mpool3', partial(F.max_pooling_2d, ksize=2, stride=2))]
        net += [('fc4', L.Linear(None, 84))]
        net += [('_sigm4', F.sigmoid)]
        net += [('fc5', L.Linear(84, 10))]
        net += [('_sigm5', F.sigmoid)]
        with self.init_scope():
            for n in net:
                if not n[0].startswith('_'):
                    setattr(self, n[0], n[1])
        self.forward = net
    def __call__(self, x):
        for n, f in self.forward:
            if not n.startswith('_'):
                x = getattr(self, n)(x)
            else:
                x = f(x)
        if chainer.config.train:
            return x
        return F.softmax(x)



class Alexnet(chainer.Chain):
    insize = 128
    def __init__(self):
        super(Alexnet, self).__init__(
            conv1=L.Convolution2D(None, 96, 11, stride=2),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 384, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            my_fc6=L.Linear(None, 100),
            my_fc7=L.Linear(None, 10),
            my_fc8=L.Linear(None, 2),
        )
        self.train = True
        

    def __call__(self, x, t):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        # h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc6(h)), chainer.using_config('train', True))
        h = F.dropout(F.relu(self.my_fc7(h)), chainer.using_config('train', True))
        h = self.my_fc8(h)
        loss = F.softmax_cross_entropy(h, t)
        chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        return loss

    def predictor(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        h = self.my_fc8(h)

        return h

gpu_id = -1 # Set to -1 if you use CPU

batchsize=8

train_iter=iterators.SerialIterator(train,batchsize)
test_iter=iterators.SerialIterator(val,batchsize,False,False)



model = resnet50.ResNet50() 
#model=alexLike.FromCaffeAlexnet()





################################################
if gpu_id >= 0:
    model.to_cpu(gpu_id)
max_epoch=100
#model=L.Classifier(LeNet5())


optimizer=optimizers.MomentumSGD(lr=0.0001)#

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
print('123')
trainer.run()
