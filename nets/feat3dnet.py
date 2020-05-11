import torch
import torch.nn as nn
import sys
sys.path.append("..")

import hyperparams as hyp
import archs.encoder3d
import utils.geom
import utils.misc
import utils.basic
import torch.nn.functional as F

EPS = 1e-4
class Feat3dNet(nn.Module):
    def __init__(self, in_dim=1):
        super(Feat3dNet, self).__init__()

        self.net = archs.encoder3d.Skipnet3d(in_dim=in_dim, out_dim=hyp.feat3d_dim).cuda()
        # self.net = archs.encoder3d.Resnet3d(in_dim=in_dim, out_dim=hyp.feat3d_dim).cuda()
        
        print(self.net)

    def forward(self, feat_input, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()
        B, C, Z, Y, X = list(feat_input.shape)

        feat = self.net(feat_input)
        feat = utils.basic.l2_normalize(feat, dim=1)
        
        if summ_writer is not None:
            summ_writer.summ_feat('feat3d/feat_input', feat_input, pca=(C>3))
            summ_writer.summ_feat('feat3d/feat_output', feat, pca=True)
    
        return total_loss, feat

