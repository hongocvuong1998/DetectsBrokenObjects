from Header import *
class MLP(Chain):     #first Model
    def __init__(self, n_mid_units=100, n_out=2):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_mid_units)
            self.l2 = L.Linear(None, n_mid_units)
            self.l3 = L.Linear(None, n_mid_units)
            self.l4 = L.Linear(None, n_mid_units)
            self.l5 = L.Linear(None, n_mid_units)
            self.l6 = L.Linear(None, n_mid_units)
            self.l7 = L.Linear(None, n_mid_units)
            self.l8 = L.Linear(None, n_out)
    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        #print('h1')
        h2 = F.relu(self.l2(h1))
        #print('h2')
        h3 = F.relu(self.l3(h2))
        #print('h3')
        h4 = F.relu(self.l4(h3))
        #print('h4')
        h5 = F.relu(self.l5(h4))
        #print('h5')
        h6 = F.relu(self.l6(h5))
        #print('h6')
        h7 = F.relu(self.l7(h6))
        #print('h7')
        return self.l8(h7)

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

class LogisticRegressionModel(Chain):

    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        with self.init_scope():
            self.w = chainer.Parameter(initializer=chainer.initializers.Normal())
            self.w.initialize([3, 1])

    def __call__(self, x, t):
        # Call the loss function
        return self.loss(x, t)

    def predictor(self, x):
        # Predict given an input (a, b, 1)
        z = F.matmul(x, self.w)
        return 1. / (1. + F.exp(-z))

    def loss(self, x, t):
        # Compute the loss for a given input (a, b, 1) and target
        y = self.predictor(x)
        loss = -t * F.log(y) - (1 - t) * F.log(1 - y)
        reporter_module.report({'loss': loss.data[0, 0]}, self)
        reporter_module.report({'w': self.w[0, 0]}, self)
        return loss

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
        

    #def __call__(self, x, t):
    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        # h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc6(h)), chainer.using_config('train', True))
        h = F.dropout(F.relu(self.my_fc7(h)), chainer.using_config('train', True))
        h = self.my_fc8(h)
        return h
        # loss = F.softmax_cross_entropy(h, t)
        # print('type loss', type(loss))
        # chainer.report({'loss': loss, 'accuracy': F.accuracy(h, t)}, self)
        # return loss

    # def predictor(self, x):
    #     h = F.max_pooling_2d(F.local_response_normalization(
    #         F.relu(self.conv1(x))), 3, stride=2)
    #     h = F.max_pooling_2d(F.local_response_normalization(
    #         F.relu(self.conv2(h))), 3, stride=2)
    #     h = F.relu(self.conv3(h))
    #     h = F.relu(self.conv4(h))
    #     h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
    #     h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
    #     h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
    #     h = self.my_fc8(h)

    #     return h
