import chainer
import chainer.functions as F
import chainer.links as L
import math

class Generator(chainer.Chain):
    def __init__(self,nz):
        self.nz = nz
        super(Generator,self).__init__(
            lz = L.Linear(self.nz, 6*6*512, wscale=0.02*math.sqrt(self.nz)),
            dc1 = L.Deconvolution2D(512, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*512)),
            dc2 = L.Deconvolution2D(256, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
            dc3 = L.Deconvolution2D(128, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
            dc4 = L.Deconvolution2D(64, 3, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
            bn01 = L.BatchNormalization(6*6*512),
            bn0 = L.BatchNormalization(512),
            bn1 = L.BatchNormalization(256),
            bn2 = L.BatchNormalization(128),
            bn3 = L.BatchNormalization(64),
        )

    def __call__(self, z, test=False):
        h = F.reshape(F.relu(self.bn01(self.lz(z), test=test)),(z.data.shape[0],512,6,6))
        h = F.relu(self.bn1(self.dc1(h), test=test))
        h = F.relu(self.bn2(self.dc2(h), test=test))
        h = F.relu(self.bn3(self.dc3(h), test=test))
        x = (self.dc4(h))
        return x

class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator,self).__init__(
        c0 = L.Convolution2D(3, 64, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*3)),
        c1 = L.Convolution2D(64, 128, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*64)),
        c2 = L.Convolution2D(128, 256, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*128)),
        c3 = L.Convolution2D(256, 512, 4, stride=2, pad=1, wscale=0.02*math.sqrt(4*4*256)),
        bn0 = L.BatchNormalization(64),
        bn1 = L.BatchNormalization(128),
        bn2 = L.BatchNormalization(256),
        bn3 = L.BatchNormalization(512),
        ll = L.Linear(6*6*512, 2, wscale=0.02*math.sqrt(6*6*512)),
        )

    def __call__(self, x, test=False):
        h = F.elu(self.bn0(self.c0(h), test=test))
        h = F.elu(self.bn1(self.c1(h), test=test))
        h = F.elu(self.bn2(self.c2(h), test=test))
        h = F.elu(self.bn3(self.c3(h), test=test))
        l = self.ll(h)
