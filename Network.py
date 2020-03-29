import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
import mxnet.ndarray as F
from mxnet.gluon.block import HybridBlock

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
#Helper
class ResnetBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.HybridSequential()
        with self.name_scope():
            self.conv_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('prelu'),
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )

    def hybrid_forward(self, F, x,*args, **kwargs):
        out = self.conv_block(x)
        return out + x

class SubpixelBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(SubpixelBlock, self).__init__()
        self.upsampling2d = nn.Conv2DTranspose(256, kernel_size=3, strides=1, padding=1)
        self.prelu = nn.Activation('prelu')

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.upsampling2d(x)
        x = self.prelu(x)
        return x

class Generator(gluon.nn.HybridBlock):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2D(64, kernel_size=9, strides=1, padding=4, activation='prelu')
        self.res_block = nn.HybridSequential()
        # Using 16 Residual Blocks
        for index in range(16):
            self.res_block.add(ResnetBlock())
        self.res_block.add(
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1, use_bias=False),
            nn.BatchNorm()
        )

        #Subpixel block
        self.subpix_block1 = SubpixelBlock()
        self.subpix_block2 = SubpixelBlock()

        self.conv2 = nn.Conv2D(3, kernel_size=9, strides=1, padding=4, activation='tanh')

    def hybrid_forward(self, F, x):
        out_conv1 = self.conv1(x)
        out_res = self.res_block(out_conv1)
        out_res_conv1 = out_conv1 + out_res
        out_sub1 = self.subpix_block1(out_res_conv1)
        out_sub2 = self.subpix_block2(out_sub1)
        output = self.conv2(out_sub2)
        return output

# Network Architecture is same as given in Paper https://arxiv.org/pdf/1609.04802.pdf
#Helper
class Discriminator_block(gluon.nn.HybridBlock):
    def __init__(self, filter, kernel_size, strides, padding):
        super(Discriminator_block, self).__init__()
        self.discriminator_block = nn.HybridSequential()
        self.discriminator_block.add(
            nn.Conv2D(filter, kernel_size=kernel_size, strides=strides, padding=padding),
            nn.BatchNorm(),
            nn.LeakyReLU(0.2)
        )
    def hybrid_forward(self, F, x):
        x = self.discriminator_block(x)
        return x

class Discriminator(gluon.nn.HybridBlock):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.HybridSequential()
        self.discriminator.add(
            nn.Conv2D(64, kernel_size=3, strides=1, padding=1),
            nn.LeakyReLU(0.2),
            Discriminator_block(filter=64, kernel_size=3, strides=2, padding=1),
            Discriminator_block(filter=128, kernel_size=3, strides=1, padding=1),
            Discriminator_block(filter=128, kernel_size=3, strides=2, padding=1),
            Discriminator_block(filter=256, kernel_size=3, strides=1, padding=1),
            Discriminator_block(filter=256, kernel_size=3, strides=2, padding=1),
            Discriminator_block(filter=512, kernel_size=3, strides=1, padding=1),
            Discriminator_block(filter=512, kernel_size=3, strides=2, padding=1),
            nn.Flatten(),
            nn.Dense(1024),
            nn.LeakyReLU(0.2),
            nn.Dense(1),
            nn.Activation('sigmoid')
        )
    def hybrid_forward(self, F, x):
        x = self.discriminator(x)
        return x