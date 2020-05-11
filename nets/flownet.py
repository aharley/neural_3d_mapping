import torch
import torch.nn as nn
import torch.nn.functional as F
# from spatial_correlation_sampler import SpatialCorrelationSampler
import numpy as np

# import sys
# sys.path.append("..")

import archs.encoder3D
import hyperparams as hyp
import utils_basic
import utils_improc
import utils_misc
import utils_samp
import math

class FlowNet(nn.Module):
    def __init__(self):
        super(FlowNet, self).__init__()

        print('FlowNet...')

        self.debug = False
        # self.debug = True
        
        self.heatmap_size = hyp.flow_heatmap_size
        # self.scales = [0.0625, 0.125, 0.25, 0.5, 0.75, 1.0]
        # self.scales = [1.0]
        # self.scales = [0.25, 0.5, 1.0]
        # self.scales = [0.125, 0.25, 0.5, 0.75, 1.0]
        self.scales = [0.25, 0.5, 0.75, 1.0]
        self.num_scales = len(self.scales)

        # self.compress_dim = 16
        # self.compressor = nn.Sequential(
        #     nn.Conv3d(in_channels=hyp.feat_dim, out_channels=self.compress_dim, kernel_size=1, stride=1, padding=0),
        # )

        self.correlation_sampler = SpatialCorrelationSampler(
            kernel_size=1,
            patch_size=self.heatmap_size,
            stride=1,
            padding=0,
            dilation_patch=1,
        ).cuda()
        
        self.flow_predictor = nn.Sequential(
            nn.Conv3d(in_channels=(self.heatmap_size**3), out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv3d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
        ).cuda()
        
        self.smoothl1 = torch.nn.SmoothL1Loss(reduction='none')
        self.smoothl1_mean = torch.nn.SmoothL1Loss(reduction='mean')
        self.mse = torch.nn.MSELoss(reduction='none')
        self.mse_mean = torch.nn.MSELoss(reduction='mean')

        print(self.flow_predictor)

    def generate_flow(self, feat0, feat1, sc):
        B, C, D, H, W = list(feat0.shape)
        utils_basic.assert_same_shape(feat0, feat1)

        if self.debug:
            print('scale = %.2f' % sc)
            print('inputs:')
            print(feat0.shape)
            print(feat1.shape)

        if not sc==1.0:
            # assert(sc==0.5 or sc==0.25) # please only use 0.25, 0.5, or 1.0 right now
            feat0 = F.interpolate(feat0, scale_factor=sc, mode='trilinear', align_corners=False)
            feat1 = F.interpolate(feat1, scale_factor=sc, mode='trilinear', align_corners=False)
            D, H, W = int(D*sc), int(H*sc), int(W*sc)
            if self.debug:
                print('downsamps:')
                print(feat0.shape)
                print(feat1.shape)

        feat0 = feat0.contiguous()
        feat1 = feat1.contiguous()

        cc = self.correlation_sampler(feat0, feat1)
        if self.debug:
            print('cc:')
            print(cc.shape)
        cc = cc.view(B, self.heatmap_size**3, D, H, W)

        cc = F.relu(cc) # relu works better than leaky relu here
        if self.debug:
            print(cc.shape)
        cc = utils_basic.l2_normalize(cc, dim=1)

        flow = self.flow_predictor(cc)
        if self.debug:
            print('flow:')
            print(flow.shape)

        if not sc==1.0:
            # note 1px here means 1px/sc at the real scale
            # first let's put the pixels in the right places
            flow = F.interpolate(flow, scale_factor=(1./sc), mode='trilinear', align_corners=False)
            # now let's correct the scale
            flow = flow/sc

        if self.debug:
            print('flow up:')
            print(flow.shape)
            
        return flow

    def forward(self, feat0, feat1, flow_g, mask_g, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        B, C, D, H, W = list(feat0.shape)
        utils_basic.assert_same_shape(feat0, feat1)

        # feats = torch.cat([feat0, feat1], dim=0)
        # feats = self.compressor(feats)
        # feats = utils_basic.l2_normalize(feats, dim=1)
        # feat0, feat1 = feats[:B], feats[B:]

        flow_total = torch.zeros_like(flow_g)
        feat1_aligned = feat1.clone()

        # summ_writer.summ_feats('flow/feats_aligned_%.2f' % 0.0, [feat0, feat1_aligned])
        feat_diff = torch.mean(utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True))
        utils_misc.add_loss('flow/feat_align_diff_%.2f' % 0.0, 0, feat_diff, 0, summ_writer)

        for sc in self.scales:
            flow = self.generate_flow(feat0, feat1_aligned, sc)
            flow_total = flow_total + flow

            # compositional LK: warp the original thing using the cumulative flow
            feat1_aligned = utils_samp.backwarp_using_3D_flow(feat1, flow_total)
            valid1_region = utils_samp.backwarp_using_3D_flow(torch.ones_like(feat1[:,0:1]), flow_total)
            # summ_writer.summ_feats('flow/feats_aligned_%.2f' % sc, [feat0, feat1_aligned],
            #                        valids=[torch.ones_like(valid1_region), valid1_region])
            feat_diff = utils_basic.reduce_masked_mean(
                utils_basic.l2_on_axis((feat1_aligned-feat0), 1, keepdim=True), valid1_region)
            utils_misc.add_loss('flow/feat_align_diff_%.2f' % sc, 0, feat_diff, 0, summ_writer)

        # ok done inference
        # now for losses/metrics:
        l1_diff_3chan = self.smoothl1(flow_total, flow_g)
        l1_diff = torch.mean(l1_diff_3chan, dim=1, keepdim=True)
        l2_diff_3chan = self.mse(flow_total, flow_g)
        l2_diff = torch.mean(l2_diff_3chan, dim=1, keepdim=True)

        nonzero_mask = ((torch.sum(torch.abs(flow_g), axis=1, keepdim=True) > 0.01).float())*mask_g
        yeszero_mask = (1.0-nonzero_mask)*mask_g
        l1_loss = utils_basic.reduce_masked_mean(l1_diff, mask_g)
        l2_loss = utils_basic.reduce_masked_mean(l2_diff, mask_g)
        l1_loss_nonzero = utils_basic.reduce_masked_mean(l1_diff, nonzero_mask)
        l1_loss_yeszero = utils_basic.reduce_masked_mean(l1_diff, yeszero_mask)
        l1_loss_balanced = (l1_loss_nonzero + l1_loss_yeszero)*0.5
        l2_loss_nonzero = utils_basic.reduce_masked_mean(l2_diff, nonzero_mask)
        l2_loss_yeszero = utils_basic.reduce_masked_mean(l2_diff, yeszero_mask)
        l2_loss_balanced = (l2_loss_nonzero + l2_loss_yeszero)*0.5

        clip = np.squeeze(torch.max(torch.abs(torch.mean(flow_g[0], dim=0))).detach().cpu().numpy()).item()
        if summ_writer is not None:
            summ_writer.summ_3D_flow('flow/flow_e_%.2f' % sc, flow_total*mask_g, clip=clip)
            summ_writer.summ_3D_flow('flow/flow_g_%.2f' % sc, flow_g, clip=clip)

        utils_misc.add_loss('flow/l1_loss_nonzero', 0, l1_loss_nonzero, 0, summ_writer)
        utils_misc.add_loss('flow/l1_loss_yeszero', 0, l1_loss_yeszero, 0, summ_writer)
        utils_misc.add_loss('flow/l1_loss_balanced', 0, l1_loss_balanced, 0, summ_writer)

        total_loss = utils_misc.add_loss('flow/l1_loss', total_loss, l1_loss, hyp.flow_l1_coeff, summ_writer)
        total_loss = utils_misc.add_loss('flow/l2_loss', total_loss, l2_loss, hyp.flow_l2_coeff, summ_writer)

        total_loss = utils_misc.add_loss('flow/warp', total_loss, feat_diff, hyp.flow_warp_coeff, summ_writer)

        # smooth loss
        dx, dy, dz = utils_basic.gradient3D(flow_total, absolute=True)
        smooth_vox = torch.mean(dx+dy+dx, dim=1, keepdims=True)
        if summ_writer is not None:
            summ_writer.summ_oned('flow/smooth_loss', torch.mean(smooth_vox, dim=3))
        smooth_loss = torch.mean(smooth_vox)
        total_loss = utils_misc.add_loss('flow/smooth_loss', total_loss, smooth_loss, hyp.flow_smooth_coeff, summ_writer)
    
        return total_loss, flow_total

