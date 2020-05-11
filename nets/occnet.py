import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import hyperparams as hyp
from utils.basic import *
import utils.improc
import utils.misc
import utils.basic

class OccNet(nn.Module):
    def __init__(self):
        super(OccNet, self).__init__()

        print('OccNet...')
        self.conv3d = nn.Conv3d(in_channels=hyp.feat3d_dim, out_channels=1, kernel_size=1, stride=1, padding=0).cuda()
        
    def compute_loss(self, pred, occ, free, valid, summ_writer):
        pos = occ.clone()
        neg = free.clone()

        # occ is B x 1 x Z x Y x X

        label = pos*2.0 - 1.0
        a = -label * pred
        b = F.relu(a)
        loss = b + torch.log(torch.exp(-b)+torch.exp(a-b))

        mask_ = (pos+neg>0.0).float()
        loss_vis = torch.mean(loss*mask_*valid, dim=3)
        summ_writer.summ_oned('occ/prob_loss', loss_vis)

        pos_loss = reduce_masked_mean(loss, pos*valid)
        neg_loss = reduce_masked_mean(loss, neg*valid)

        balanced_loss = pos_loss + neg_loss

        return balanced_loss

    def forward(self, feat, occ_g, free_g, valid, summ_writer, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        occ_e_ = self.conv3d(feat)
        # occ_e_ is B x 1 x Z x Y x X
        
        # smooth loss
        dz, dy, dx = gradient3d(occ_e_, absolute=True)
        smooth_vox = torch.mean(dx+dy+dz, dim=1, keepdims=True)
        summ_writer.summ_oned('occ/smooth_loss%s' % suffix, torch.mean(smooth_vox, dim=3))
        smooth_loss = utils.basic.reduce_masked_mean(smooth_vox, valid)
        total_loss = utils.misc.add_loss('occ/smooth_loss%s' % suffix, total_loss, smooth_loss, hyp.occ_smooth_coeff, summ_writer)
    
        occ_e = F.sigmoid(occ_e_)
        occ_e_binary = torch.round(occ_e)

        # collect some accuracy stats 
        occ_match = occ_g*torch.eq(occ_e_binary, occ_g).float()
        free_match = free_g*torch.eq(1.0-occ_e_binary, free_g).float()
        either_match = torch.clamp(occ_match+free_match, 0.0, 1.0)
        either_have = torch.clamp(occ_g+free_g, 0.0, 1.0)
        acc_occ = reduce_masked_mean(occ_match, occ_g*valid)
        acc_free = reduce_masked_mean(free_match, free_g*valid)
        acc_total = reduce_masked_mean(either_match, either_have*valid)
        acc_bal = (acc_occ + acc_free)*0.5

        summ_writer.summ_scalar('unscaled_occ/acc_occ%s' % suffix, acc_occ.cpu().item())
        summ_writer.summ_scalar('unscaled_occ/acc_free%s' % suffix, acc_free.cpu().item())
        summ_writer.summ_scalar('unscaled_occ/acc_total%s' % suffix, acc_total.cpu().item())
        summ_writer.summ_scalar('unscaled_occ/acc_bal%s' % suffix, acc_bal.cpu().item())

        if summ_writer is not None:
            # vis
            summ_writer.summ_occ('occ/occ_g%s' % suffix, occ_g)
            summ_writer.summ_occ('occ/free_g%s' % suffix, free_g)
            summ_writer.summ_occ('occ/occ_e%s' % suffix, occ_e)
            # summ_writer.summ_occ('occ/valid%s' % suffix, valid)
        
        prob_loss = self.compute_loss(occ_e_, occ_g, free_g, valid, summ_writer)
        total_loss = utils.misc.add_loss('occ/prob_loss%s' % suffix, total_loss, prob_loss, hyp.occ_coeff, summ_writer)

        return total_loss, occ_e

