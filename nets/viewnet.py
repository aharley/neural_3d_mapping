import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.renderer
import hyperparams as hyp
from utils.basic import *
import utils.improc
import utils.basic
import utils.misc
import utils.geom

class ViewNet(nn.Module):
    def __init__(self):
        super(ViewNet, self).__init__()

        print('ViewNet...')

        self.net = archs.renderer.Net3d2d(hyp.feat3d_dim, 64, 32, hyp.view_depth, depth_pool=8).cuda()

        self.rgb_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
        ).cuda()
        self.emb_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, hyp.feat2d_dim, kernel_size=1, stride=1, padding=0),
        ).cuda()
        print(self.net)

    def forward(self, pix_T_cam0, cam0_T_cam1, feat_mem1, rgb_g, vox_util, valid=None, summ_writer=None, test=False, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        B, C, H, W = list(rgb_g.shape)

        PH, PW = hyp.PH, hyp.PW
        if (PH < H) or (PW < W):
            # print('H, W', H, W)
            # print('PH, PW', PH, PW)
            sy = float(PH)/float(H)
            sx = float(PW)/float(W)
            pix_T_cam0 = utils.geom.scale_intrinsics(pix_T_cam0, sx, sy)

            if valid is not None:
                valid = F.interpolate(valid, scale_factor=0.5, mode='nearest')
            rgb_g = F.interpolate(rgb_g, scale_factor=0.5, mode='bilinear')
            
        feat_proj = vox_util.apply_pixX_T_memR_to_voxR(
            pix_T_cam0, cam0_T_cam1, feat_mem1,
            hyp.view_depth, PH, PW)

        feat = self.net(feat_proj)
        rgb = self.rgb_layer(feat)
        emb = self.emb_layer(feat)
        emb = utils.basic.l2_normalize(emb, dim=1)

        if test:
            return None, rgb, None
        
        loss_im = utils.basic.l1_on_axis(rgb-rgb_g, 1, keepdim=True)
        if valid is not None:
            rgb_loss = utils.basic.reduce_masked_mean(loss_im, valid)
        else:
            rgb_loss = torch.mean(loss_im)

        total_loss = utils.misc.add_loss('view/rgb_l1_loss', total_loss, rgb_loss, hyp.view_l1_coeff, summ_writer)
            
        # vis
        if summ_writer is not None:
            summ_writer.summ_oned('view/rgb_loss', loss_im)
            summ_writer.summ_rgbs('view/rgb', [rgb.clamp(-0.5, 0.5), rgb_g])
            summ_writer.summ_rgb('view/rgb_e', rgb.clamp(-0.5, 0.5))
            summ_writer.summ_rgb('view/rgb_g', rgb_g.clamp(-0.5, 0.5))
            summ_writer.summ_feat('view/emb', emb, pca=True)
            if valid is not None:
                summ_writer.summ_rgb('view/rgb_e_valid', valid*rgb.clamp(-0.5, 0.5))
                summ_writer.summ_rgb('view/rgb_g_valid', valid*rgb_g.clamp(-0.5, 0.5))

        return total_loss, rgb, emb
