import numpy as np
from chainer.dataset import iterator as itr_module
from chainer import training

def clip_img(x):
    return np.float32(-1 if x<-1 else (1 if x > 1 else x))

class DCGANUpdater(training.StandardUpdater):
    def __init__(self,iterator,generator,discriminator,op_g,op_d,device):
        if isinstance(iterator, itr_module.Iterator):
            iterator = {'main':iterator}
        self._iterators = iterator
        self.generator = generator
        self.discriminator = discriminator
        self._optimizers = {'generator':op_g,'discriminator':op_d}
        self.device = device
        self.iteration = 0

    def update_core(self):
        batch = self._iterators['main'].next()
