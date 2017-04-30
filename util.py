import numpy as np
from chainer.dataset import iterator as itr_module
from chainer.dataset import convert 
from chainer import training, Variable
import chainer

def clip_img(x):
    return np.float32(-1 if x<-1 else (1 if x > 1 else x))

class DCGANUpdater(training.StandardUpdater):
    def __init__(self,iterator,generator,discriminator,op_g,op_d,device,nz):
        if isinstance(iterator, itr_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.generator = generator
        self.discriminator = discriminator
        self._optimizers = {'generator':op_g,'discriminator':op_d}
        self.converter = convert.concat_examples
        self.device = device
        self.iteration = 0
        self.nz = nz

    def update_core(self):
        batch = self._iterators['main'].next()
        images = self.converter(batch, self.device)
        batchsize = images.shape[0]
        x_real = Variable(images) / 255.
        xp = chainer.cuda.get_array_module(x_real.data)

        z = Variable(xp.asarray(np.random.uniform(-1,1,(batchsize,self.nz)).astype(np.float32)))
        x_fake = self.generator(z, test=False)

        y_real = self.discriminator(x_real,test=False)
        y_fake = self.discriminator(x_fake,test=False)

        L1 = F.sum(F.softmax_cross_entropy(y_real, Variable(xp.ones(self.batchsize).astype(np.int32))))/ batchsize
        L2 = F.sum(F.softmax_cross_entropy(y_fake, Variable(xp.ones(self.batchsize).astype(np.int32))))/ batchsize
        loss_dis = L1 + L2
        loss_gen = F.sum(F.softmax_cross_entropy(y_fake, Variable(xp.zeros(self.batchsize).astype(np.int32)))) / batchsize

        self.discriminator.cleargrads()
        loss_dis.backward()
        self.optimizers['discriminator'].update()

        self.generator.cleargrads()
        loss_gen.backward()
        self.optimizers['generator'].update()
        

        chainer.reporter.report({
            'loss/dis':loss_dis,
            'loss/gen':loss_gen,
            })




