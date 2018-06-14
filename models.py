from mxnet import gluon

class Generator(gluon.HybridBlock):
    def __init__(self,dims = 64, name = 'dcganG', verbose=False,**kwargs):
        super(Generator,self).__init__(**kwargs)
        self.dims = dims
        self.module_name = name
        self.verbose=verbose

        with self.name_scope():
            self.fc1 = gluon.nn.Dense(dims * 16 * 4 * 4)
            self.deconv1 = gluon.nn.Conv2DTranspose(dims * 8, 4, 2, 1, use_bias=False)
            self.deconv3 = gluon.nn.Conv2DTranspose(dims * 2, 4, 2, 1, use_bias=False)
            self.deconv5 = gluon.nn.Conv2DTranspose(3, 4, 2, 1, use_bias=False)

            self.fc1_bn = gluon.nn.BatchNorm()
            self.deconv1_bn = gluon.nn.BatchNorm()
            self.deconv3_bn = gluon.nn.BatchNorm()
        return
    def hybrid_forward(self, F, x):
        if self.verbose:
            print 'netG input: ',x.shape
        x = self.fc1(x)
        x = F.reshape(x, shape=[-1, self.dims * 16, 4, 4])
        x = F.relu(self.fc1_bn(x))
        x = F.relu(self.deconv1_bn(self.deconv1(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv5(x))
        if self.verbose:
            print 'netG output: ',x.shape
        return x

    def save_params(self):
        self.collect_params().save(self.module_name)
        return
    def load_params(self, ctx, allow_missing=False, ignore_extra = False):
        self.collect_params().load(self.module_name, ctx, allow_missing,ignore_extra)
        return



class Discriminator(gluon.HybridBlock):
    def __init__(self,dims = 64, name='dcganD', verbose=False,**kwargs):
        super(Discriminator,self).__init__(**kwargs)
        self.dims = dims
        self.module_name = name
        self.verbose=verbose

        self.conv1 = gluon.nn.Conv2D(dims, 6, 4, 1, use_bias=False)
        self.conv3 = gluon.nn.Conv2D(dims*2,4,2,1, use_bias=False)
        self.conv5 = gluon.nn.Conv2D(1,4,1,0, use_bias=False)

        self.bn3 = gluon.nn.BatchNorm()

        self.flatten = gluon.nn.Flatten()
        return
    def hybrid_forward(self,F,x):
        if self.verbose:
            print "netD input:",x.shape
        x = self.conv1(x)
        x = F.LeakyReLU(x,slope=0.2)
        x = self.bn3(self.conv3(x))
        x = F.LeakyReLU(x,slope=0.2)
        x = self.flatten(self.conv5(x))
        if self.verbose:
            print 'netD output:',x.shape
        return x
    def save_params(self):
        self.collect_params().save(self.module_name)
        return
    def load_params(self,ctx, allow_missing=False, ignore_extra = False):
        self.collect_params().load(self.module_name,ctx, allow_missing,ignore_extra)
        return
