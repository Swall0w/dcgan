import chainer
from chainer import cuda, optimizers, datasets
from model import Generator, Discriminator
import argparse
from PIL import Image
import numpy as np
import os

def arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch','-b',type=int,default=100,help='number of minibatch')
    parser.add_argument('--epoch','-e',type=int,default=100,help='number of epoch')
    parser.add_argument('--gpu','-g',type=int,default=-1,help='number of gpu')
    parser.add_argument('--output','-o',default='result',help='output directory')
    parser.add_argument('--resume','-r',default='',help='resume the training from snapshot')
    parser.add_argument('--unit','-u',type=int,default=500,help='number of unit')
    return parser.parse_args()

def main():
    args = arg()
    nz = 100
    image_path = 'data/'

    generator = Generator(nz)
    discriminator = Discriminator()

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        generator.to_gpu()
        discriminator.to_gpu()
        zvis = (cuda.cupy.random.uniform(-1,1,(100,nz), dtype=np.float32))
    else:
        zvis = (np.random.uniform(-1,1,(100,nz)))
        

    op_g = optimizers.Adam(alpha=0.0002, beta1=0.5)
    op_g.setup(generator)
    op_g.add_hook(chainer.optimizer.WeightDecay(0.00001))

    op_d = optimizers.Adam(alpha=0.0002, beta1=0.5)
    op_d.setup(discriminator)
    op_d.add_hook(chainer.optimizer.WeightDecay(0.00001))

    train = datasets.ImageDataset(os.listdir(image_path))


if __name__ == '__main__':
    main()
