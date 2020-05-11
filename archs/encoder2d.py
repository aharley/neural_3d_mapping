import torch
import torch.nn as nn

# import hyperparams as hyp
# from utils_basic import *

class Skipnet2d(nn.Module):
    def __init__(self, in_chans, mid_chans=64, out_chans=1):
        super(Skipnet2d, self).__init__()
        conv2d = []
        conv2d_transpose = []
        up_bn = []

        self.down_in_dims = [in_chans, mid_chans, 2*mid_chans]
        self.down_out_dims = [mid_chans, 2*mid_chans, 4*mid_chans]
        self.down_ksizes = [3, 3, 3]
        self.down_strides = [2, 2, 2]
        padding = 1

        for i, (in_dim, out_dim, ksize, stride) in enumerate(zip(self.down_in_dims, self.down_out_dims, self.down_ksizes, self.down_strides)):
            conv2d.append(nn.Sequential(
                nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=out_dim),
            ))
        self.conv2d = nn.ModuleList(conv2d)

        self.up_in_dims = [4*mid_chans, 6*mid_chans]
        self.up_bn_dims = [6*mid_chans, 3*mid_chans]
        self.up_out_dims = [4*mid_chans, 2*mid_chans]
        self.up_ksizes = [4, 4]
        self.up_strides = [2, 2]
        padding = 1 # Note: this only holds for ksize=4 and stride=2!
        print('up dims: ', self.up_out_dims)

        for i, (in_dim, bn_dim, out_dim, ksize, stride) in enumerate(zip(self.up_in_dims, self.up_bn_dims, self.up_out_dims, self.up_ksizes, self.up_strides)):
            conv2d_transpose.append(nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_dim, out_channels=out_dim, kernel_size=ksize, stride=stride, padding=padding),
                nn.LeakyReLU(),
            ))
            up_bn.append(nn.BatchNorm2d(num_features=bn_dim))

        # final 1x1x1 conv to get our desired out_chans
        self.final_feature = nn.Conv2d(in_channels=3*mid_chans, out_channels=out_chans, kernel_size=1, stride=1, padding=0)
        self.conv2d_transpose = nn.ModuleList(conv2d_transpose)
        self.up_bn = nn.ModuleList(up_bn)
        
    def forward(self, inputs):
        feat = inputs
        skipcons = []
        for conv2d_layer in self.conv2d:
            feat = conv2d_layer(feat)
            skipcons.append(feat)

        skipcons.pop() # we don't want the innermost layer as skipcon

        for i, (conv2d_transpose_layer, bn_layer) in enumerate(zip(self.conv2d_transpose, self.up_bn)):
            feat = conv2d_transpose_layer(feat)
            feat = torch.cat([feat, skipcons.pop()], dim=1) # skip connection by concatenation
            feat = bn_layer(feat)

        feat = self.final_feature(feat)

        return feat

if __name__ == "__main__":
    net = Skipnet2d(in_chans=4, mid_chans=32, out_chans=3)
    print(net.named_parameters)
    inputs = torch.rand(2, 4, 128, 384)
    out = net(inputs)
    print(out.size())


