import torch
import torch.nn as nn

class Net3d2d(nn.Module):
    def __init__(self, in_chans, mid_chans, out_chans, depth, depth_pool=8):
        super(Net3d2d, self).__init__()

        pool = []
        conv3d = []
        conv2d = []
        
        # one maxpool along depth
        pool.append(nn.MaxPool3d([depth_pool,1,1], stride=[depth_pool,1,1], padding=0, dilation=1))
        self.pool = nn.ModuleList(pool)

        # one 3d conv
        in_dims = [in_chans]
        out_dims = [mid_chans]
        ksize = 3
        stride = 1
        padding = 1
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            conv3d.append(nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding, bias=True),
                nn.LeakyReLU(),
                nn.BatchNorm3d(num_features=out_dim),
            ))
        self.conv3d = nn.ModuleList(conv3d)

        # (reshape 3d to 2d done in forward pass)
            
        # a couple 2d convs
        in_dims = [mid_chans*int(depth/depth_pool), mid_chans*2]
        out_dims = [mid_chans*2, mid_chans*4]
        ksize = 3
        stride = 1
        padding = 1
        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims)):
            conv2d.append(nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=ksize, stride=stride, padding=padding, bias=True),
                nn.LeakyReLU(),
                nn.BatchNorm2d(num_features=out_dim),
            ))
        self.conv2d = nn.ModuleList(conv2d)

        # final 1x1x1 conv to get our desired out_chans
        self.final_conv = nn.Conv2d(out_dims[-1], out_chans, kernel_size=1, stride=1, padding=0, bias=True)
        
    def forward(self, feat):
        for pool_layer in self.pool:
            feat = pool_layer(feat)
            
        for conv3d_layer in self.conv3d:
            feat = conv3d_layer(feat)
            
        # squash depth into channels
        B, C, D, H, W = list(feat.shape)
        feat = feat.view(B, C*D, H, W)
        
        for conv2d_layer in self.conv2d:
            feat = conv2d_layer(feat)
        
        feat = self.final_conv(feat)
        return feat

if __name__ == "__main__":
    in_chans = 8
    depth = 16
    net = Net3d2d(in_chans=in_chans, mid_chans=32, out_chans=32, depth=depth, depth_pool=8)
    print(net.named_parameters)
    inputs = torch.rand(2, in_chans, depth, 64, 192)
    print('running random input through...')
    out = net(inputs)
    print('done!')
    print(out.size())


