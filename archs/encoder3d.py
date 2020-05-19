import torch
import torch.nn as nn
import time
# import hyperparams as hyp
# from utils_basic import *
import torch.nn.functional as F
import archs.pixelshuffle3d
import spconv

class Skipnet3d(nn.Module):
    def __init__(self, in_dim, out_dim, chans=64):
        super(Skipnet3d, self).__init__()
        conv3d = []
        up_bn = [] # batch norm for deconv
        conv3d_transpose = []

        self.down_in_dims = [in_dim, chans, 2*chans]#, 4*chans]
        self.down_out_dims = [chans, 2*chans, 4*chans, 8*chans]
        self.down_ksizes = [4, 4, 4, 4]
        self.down_strides = [2, 2, 2, 2]
        padding = 1
        # print('down dims: ', self.down_out_dims)

        for i, (in_chan, out_chan, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            conv3d.append(nn.Sequential(
                nn.ReplicationPad3d(padding),
                nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, stride=stride, padding=0),
                # nn.Conv3d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_chan),
            ))
        self.conv3d = nn.ModuleList(conv3d)

        self.up_in_dims = [4*chans, 6*chans]
        self.up_out_dims = [4*chans, 4*chans]
        self.up_bn_dims = [6*chans, 5*chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 
        # print('up dims: ', self.up_out_dims)

        for i, (in_chan, bn_dim, out_chan, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
            conv3d_transpose.append(nn.Sequential(
                nn.ConvTranspose3d(in_channels=in_chan, out_channels=out_chan, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm3d(num_features=bn_dim))
        self.conv3d_transpose = nn.ModuleList(conv3d_transpose)
        self.up_bn = nn.ModuleList(up_bn)

        # final 1x1x1 conv to get our desired out_dim
        self.final_feature = nn.Conv3d(in_channels=self.up_bn_dims[-1], out_channels=out_dim, kernel_size=1, stride=1, padding=0)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            skipcons.append(feat)
        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv3d_transpose_layer, bn_layer) in enumerate(zip(self.conv3d_transpose, self.up_bn)):
            # print('feat before up', feat.shape)
            feat = conv3d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) #skip connection by concatenation
            # print('feat before bn', feat.shape)
            feat = bn_layer(feat)

        feat = self.final_feature(feat)
        return feat

class Res3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, padding=1):
        super(Res3dBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
            nn.Conv3d(out_planes, out_planes, kernel_size=3, stride=1, padding=padding),
            nn.BatchNorm3d(out_planes)
        )
        assert(padding==1 or padding==0)
        self.padding = padding

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm3d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        # print('res', res.shape)
        skip = self.skip_con(x)
        if self.padding==0:
            # the data has shrunk a bit
            skip = skip[:,:,2:-2,2:-2,2:-2]
        # print('skip', skip.shape)
        return F.relu(res + skip, True)

class Conv3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Conv3dBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.conv(x)

class Pool3dBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool3dBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool3d(x, kernel_size=self.pool_size, stride=self.pool_size)

class Deconv3dBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Deconv3dBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.deconv(x)

class Resnet3d(nn.Module):
    def __init__(self, in_dim, out_dim, chans=32):
        super().__init__()

        self.encoder_layer0 = Res3dBlock(in_dim, chans)
        self.encoder_layer1 = Pool3dBlock(2)
        self.encoder_layer2 = Res3dBlock(chans, chans)
        self.encoder_layer3 = Res3dBlock(chans, chans)
        self.encoder_layer4 = Res3dBlock(chans, chans)
        self.encoder_layer5 = Pool3dBlock(2)
        self.encoder_layer6 = Res3dBlock(chans, chans)
        self.encoder_layer7 = Res3dBlock(chans, chans)
        self.encoder_layer8 = Res3dBlock(chans, chans)
        self.encoder_layer9 = Deconv3dBlock(chans, chans)
        self.final_layer = nn.Conv3d(in_channels=chans, out_channels=out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.encoder_layer0(x)
        x = self.encoder_layer1(x)
        x = self.encoder_layer2(x)
        x = self.encoder_layer3(x)
        x = self.encoder_layer4(x)
        x = self.encoder_layer5(x)
        x = self.encoder_layer6(x)
        x = self.encoder_layer7(x)
        x = self.encoder_layer8(x)
        x = self.encoder_layer9(x)
        x = self.final_layer(x)
        return x
    
    
    



    
