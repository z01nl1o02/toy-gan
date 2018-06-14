import os,pdb
import matplotlib.pyplot as plt
import time, math, argparse

import mxnet as mx
import mxnet.ndarray as nd
from mxnet import autograd
import numpy as np
from models import Generator,Discriminator
from datasetX import DatasetX
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--epochs','-e',help='No. of epoch to run for training',type=int,default=100)
parser.add_argument('--resume','-c',help='Continue training? 1:Yes, 0:False',type=int,default=0)
args = parser.parse_args()

def time_since(start):
    now = time.time()
    s=now-start
    m=math.floor(s/60)
    s-=m*60
    return '%dm %ds'%(m,s)

data_path = "data/mnist/"

epochs=10
batch_size=16
z_dims=100
img_dims=32

use_gpu=True
ctx=mx.gpu() if use_gpu else mx.cpu()

lr=0.0002
beta1=0.5
beta2=0.999

def visualize(img_arr,idx):
    img = ((img_arr.asnumpy().transpose(1,2,0) + 1.0)*127.5).astype(np.uint8)
    if idx >= 0:
       cv2.imwrite('%d.jpg'%idx,img)
    else:
        plt.imshow(img)
        plt.axis('off')
    return

def save_netG_output(netG,idx):
    rows = 10
    cols = 10
    canvas = np.zeros((rows*img_dims,cols*img_dims,3))
    latent_z=mx.nd.random_normal(0,1,shape=(rows * cols,z_dims),ctx=ctx)
    fakes=netG(latent_z)
    for k in range(fakes.shape[0]):
        #pdb.set_trace()
        img = ((fakes[k].asnumpy().transpose(1,2,0) + 1.0)*127.5).astype(np.uint8)
        row = k // cols
        col = k - row * cols
        canvas[row*img_dims:(row+1)*img_dims, col * img_dims:(col+1)*img_dims,:] = img 
    canvas = np.uint8(canvas)
    cv2.imwrite('netG_%d.jpg'%idx,canvas)
    return 
        
    
    
    
netG=Generator(name="dcganG")
netD=Discriminator(name="dcganD")

loss=mx.gluon.loss.SigmoidBinaryCrossEntropyLoss()

real_label=nd.ones((batch_size,),ctx=ctx)
fake_label=nd.zeros((batch_size,),ctx=ctx)

img_list=[os.path.join(data_path,x) for x in os.listdir(data_path) if x.endswith('jpg')]
train_data=DatasetX(img_list,img_dims,batch_size=batch_size)

def init_params():
    netG.initialize(mx.initializer.Normal(0.02),ctx=ctx)
    netD.initialize(mx.initializer.Normal(0.02),ctx=ctx)

def load_weights():
    netG.load_params(ctx=ctx)
    netD.load_params(ctx=ctx)

def init_optimizers():
    trainerG=mx.gluon.Trainer(netG.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1,'beta2':beta2})
    trainerD=mx.gluon.Trainer(netD.collect_params(),'adam',{'learning_rate':lr,'beta1':beta1,'beta2':beta2})
    return trainerG, trainerD

def facc(label, pred):
    pred=pred.ravel()
    label=label.ravel()
    return ((pred>0.5)==label).mean()

metric=mx.metric.CustomMetric(facc)

def train(epochs, resume=False):
    if resume:
        load_weights()
        print('loading pretrained weights')
    else:
        init_params()
    trainerG, trainerD=init_optimizers()
    netD.hybridize()
    netG.hybridize()
   #netD.initialize(ctx=ctx)
   # netG.initialize(ctx=ctx)
    print('training for %d epochs...'%epochs)

    start=time.time()
    roundNum = 0
    for epoch in range(epochs):
        train_data.reset()
        iteration = 0
        while train_data.has_next():
            roundNum += 1
            #update D: max logD(x) + log(1-D(G(z)))
            data=train_data.next().as_in_context(ctx)
            latent_z=mx.nd.random_normal(0,1,shape=(batch_size,z_dims),ctx=ctx)
            with autograd.record():
                #train with real image
                output=netD(data)
                errD_real=loss(output,real_label)
                metric.update([real_label,],[output,])
                #train with fake image
                fake=netG(latent_z)
                output=netD(fake.detach())
                errD_fake=loss(output,fake_label)
                errD=errD_real+errD_fake
                errD.backward()
                metric.update([fake_label,],[output,])
            trainerD.step(batch_size)
            
            #update G : max log(D(G(z))
            with autograd.record():
                output=netD(fake)
                errG=loss(output,real_label)
                errG.backward()
            trainerG.step(batch_size)
            if iteration%100 == 0:
                _,acc=metric.get()
                print('epoch {%d}: iter {%d} dloss={%f},generator loss={%f},training acc={%f}'%(epoch,iteration,nd.mean(errD).asscalar(), nd.mean(errG).asscalar(),acc))
                save_netG_output(netG,roundNum)
                #plt.show()
            iteration += 1
        name,acc = metric.get()
        print('epoch {%d}: iter {%d} dloss={%f},generator loss={%f},training acc={%f}'%(epoch,iteration,nd.mean(errD).asscalar(), nd.mean(errG).asscalar(),acc))
        print('time:{%s}'%time_since(start))
        metric.reset()
        netG.save_params()
        netD.save_params()
    print('time:{%s}'%time_since(start))
    netG.save_params()
    netD.save_params()

train(args.epochs, bool(args.resume))

