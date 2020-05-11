# Reference: https://github.com/dragonbook/V2V-PoseNet-pytorch

import torch.nn as nn
import torch.nn.functional as F


class Basic2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(Basic2DBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=1, padding=((kernel_size-1)//2)),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class Res2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Res2DBlock, self).__init__()
        self.res_branch = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_planes)
        )

        if in_planes == out_planes:
            self.skip_con = nn.Sequential()
        else:
            self.skip_con = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(out_planes)
            )

    def forward(self, x):
        res = self.res_branch(x)
        skip = self.skip_con(x)
        return F.relu(res + skip, True)


class Pool2DBlock(nn.Module):
    def __init__(self, pool_size):
        super(Pool2DBlock, self).__init__()
        self.pool_size = pool_size

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.pool_size)


class Upsample2DBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(Upsample2DBlock, self).__init__()
        assert(kernel_size == 2)
        assert(stride == 2)
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=0, output_padding=0),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderDecoder(nn.Module):
    def __init__(self, in_dim=32, scale=0.125):
        super().__init__()

        self.encoder_pool1 = Pool2DBlock(2)
        self.encoder_res1 = Res2DBlock(in_dim, 64)
        self.encoder_pool2 = Pool2DBlock(2)
        self.encoder_res2 = Res2DBlock(64, 128)
        self.encoder_pool3 = Pool2DBlock(2)
        self.encoder_res3 = Res2DBlock(128, 128)
        self.encoder_pool4 = Pool2DBlock(2)
        self.encoder_res4 = Res2DBlock(128, 128)
        self.encoder_pool5 = Pool2DBlock(2)
        self.encoder_res5 = Res2DBlock(128, 128)

        self.mid_res = Res2DBlock(128, 128)

        self.decoder_res5 = Res2DBlock(128, 128)
        self.decoder_upsample5 = Upsample2DBlock(128, 128, 2, 2)
        self.decoder_res4 = Res2DBlock(128, 128)
        self.decoder_upsample4 = Upsample2DBlock(128, 128, 2, 2)
        self.decoder_res3 = Res2DBlock(128, 128)
        self.decoder_upsample3 = Upsample2DBlock(128, 128, 2, 2)
        self.decoder_res2 = Res2DBlock(128, 128)
        self.decoder_upsample2 = Upsample2DBlock(128, 64, 2, 2)
        # self.decoder_res1 = Res2DBlock(64, 64)
        # self.decoder_upsample1 = Upsample2DBlock(64, 32, 2, 2)

        self.skip_res1 = Res2DBlock(in_dim, 32)
        self.skip_res2 = Res2DBlock(64, 64)
        self.skip_res3 = Res2DBlock(128, 128)
        self.skip_res4 = Res2DBlock(128, 128)
        self.skip_res5 = Res2DBlock(128, 128)

    def forward(self, x):
        # skip_x1 = self.skip_res1(x)
        x = self.encoder_pool1(x)
        x = self.encoder_res1(x)
        # skip_x2 = self.skip_res2(x)
        x = self.encoder_pool2(x)
        x = self.encoder_res2(x)
        skip_x3 = self.skip_res3(x)
        x = self.encoder_pool3(x)
        x = self.encoder_res3(x)
        skip_x4 = self.skip_res4(x)
        x = self.encoder_pool4(x)
        x = self.encoder_res4(x)
        skip_x5 = self.skip_res5(x)
        x = self.encoder_pool5(x)
        x = self.encoder_res5(x)

        x = self.mid_res(x)

        x = self.decoder_res5(x)
        x = self.decoder_upsample5(x)
        x = x + skip_x5
        x = self.decoder_res4(x)
        x = self.decoder_upsample4(x)
        # end here for 1/8 res
        
        x = x + skip_x4
        x = self.decoder_res3(x)
        x = self.decoder_upsample3(x)
        # end here for 1/4 res
        
        # x = x + skip_x3
        # x = self.decoder_res2(x)
        # x = self.decoder_upsample2(x)
        # # end here for 1/2 res
        
        # x = x + skip_x2
        # x = self.decoder_res1(x)
        # x = self.decoder_upsample1(x)
        # x = x + skip_x1
        # end here for 1/1 res

        return x


class V2VModel(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()

        # self.front_layers = nn.Sequential(
        #     Basic2DBlock(input_channels, 16, 7),
        #     Res2DBlock(16, 32),
        #     Res2DBlock(32, 32),
        #     Res2DBlock(32, 32)
        # )

        # self.encoder_decoder = EncoderDecoder()
        self.encoder_decoder = EncoderDecoder(in_dim=input_channels)

        # self.back_layers = nn.Sequential(
        #     Res2DBlock(32, 32),
        #     Basic2DBlock(32, 32, 1),
        #     Basic2DBlock(32, 32, 1),
        # )

        # self.output_layer = nn.Conv2d(32, output_channels, kernel_size=1, stride=1, padding=0) # full res
        # self.output_layer = nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0) # 1/2 res
        self.output_layer = nn.Conv2d(128, output_channels, kernel_size=1, stride=1, padding=0) # 1/4 res
        # self.output_layer = nn.Conv2d(128, output_channels, kernel_size=1, stride=1, padding=0) # 1/8 res

        self._initialize_weights()

    def forward(self, x):
        # x = self.front_layers(x)
        x = self.encoder_decoder(x)
        # x = self.back_layers(x)
        x = self.output_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)
