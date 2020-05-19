import numpy as np
import hyperparams as hyp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.ops as ops

import utils.basic
import utils.improc
import utils.geom
import utils.misc
import utils.samp

EPS = 1e-6

def eval_against_gt(loss, cam0_T_cam1_e, cam0_T_cam1_g,
                    t_coeff=0.0, deg_coeff=0.0, sc=1.0,
                    summ_writer=None):
    # cam0_T_cam1_e is B x 4 x 4
    # cam0_T_cam1_g is B x 4 x 4
    r_e, t_e = utils.geom.split_rt(cam0_T_cam1_e)
    r_g, t_g = utils.geom.split_rt(cam0_T_cam1_g)
    _, ry_e, _ = utils.geom.rotm2eul(r_e)
    _, ry_g, _ = utils.geom.rotm2eul(r_g)
    deg_e = torch.unsqueeze(utils.geom.rad2deg(ry_e), axis=-1)
    deg_g = torch.unsqueeze(utils.geom.rad2deg(ry_g), axis=-1)

    t_l2 = torch.mean(utils.basic.sql2_on_axis(t_e-t_g, 1))
    loss = utils.misc.add_loss('t_sql2_%.2f' % sc,
                               loss,
                               t_l2,
                               t_coeff,
                               summ_writer=summ_writer)

    deg_l2 = torch.mean(utils.basic.sql2_on_axis(deg_e-deg_g, 1))
    loss = utils.misc.add_loss('deg_sql2_%.2f' % sc,
                               loss,
                               deg_l2,
                               deg_coeff,
                               summ_writer=summ_writer)

    return loss

def cost_volume_3D(vox0, vox1,
                   max_disp_z=4,
                   max_disp_y=1,
                   max_disp_x=4):
    # max_disp = max_displacement
    # vox0 is B x C x Z x Y x X 
    # vox1 is B x C x Z x Y x X 
    # return cost_vol, shaped B x E x Z x Y x X
    # E_i = max_disp_i*2 + 1
    # E = \prod E_i

    # pad the top, bottom, left, and right of vox1
    ones = torch.ones_like(vox1)
    vox1_pad = F.pad(vox1,
                     (max_disp_z, max_disp_z,
                      max_disp_y, max_disp_y,
                      max_disp_x, max_disp_x),
                     'constant', 0)
    
    ones_pad = F.pad(ones,
                     (max_disp_z, max_disp_z,
                      max_disp_y, max_disp_y,
                      max_disp_x, max_disp_x),
                     'constant', 0)

    _, _, d, h, w = vox0.shape
    loop_range1 = max_disp_z * 2 + 1
    loop_range2 = max_disp_y * 2 + 1
    loop_range3 = max_disp_x * 2 + 1
    cost_vol = []
    for z in range(0, loop_range1):
        for y in range(0, loop_range2):
            for x in range(0, loop_range3):
                vox1_slice = vox1_pad[:, :, z:z+d, y:y+h, x:x+w]
                ones_slice = ones_pad[:, :, z:z+d, y:y+h, x:x+w]
                cost = utils.basic.reduce_masked_mean(vox0*vox1_slice, ones_slice, dim=1, keepdim=True)
                cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, axis=1)
    return cost_vol

class EgoNet(nn.Module):
    def __init__(self,
                 num_scales=1,
                 num_rots=3,
                 max_deg=4,
                 max_disp_z=1,
                 max_disp_y=1,
                 max_disp_x=1):
        print('EgoNet...')
        super(EgoNet, self).__init__()
        if num_scales:
            self.scales = [1]
        elif num_scales==2:
            self.scales = [0.5, 1]
        else:
            assert(False) # only 1-2 scales supported right now
        self.R = num_rots
        self.max_deg = max_deg # max degrees rotation, on either side of zero
        self.max_disp_z = max_disp_z
        self.max_disp_y = max_disp_y
        self.max_disp_x = max_disp_x
        
        self.E1 = self.max_disp_z*2 + 1
        self.E2 = self.max_disp_y*2 + 1
        self.E3 = self.max_disp_x*2 + 1
        self.E = self.E1*self.E2*self.E3

        self.first_layer = nn.Linear(self.R*self.E, 128).cuda()
        self.second_layer = nn.Linear(128, 128).cuda()
        self.third_layer = nn.Linear(128, 4).cuda()

    def forward(self, feat0, feat1, cam0_T_cam1_g, vox_util, summ_writer, reuse=False):
        total_loss = 0.0

        utils.basic.assert_same_shape(feat0, feat1)

        summ_writer.summ_feats('ego/feats', [feat0, feat1], pca=True)

        total_loss, cam0_T_cam1_e, feat1_warped = self.multi_scale_corr3Dr(
            total_loss, feat0, feat1, vox_util, summ_writer, cam0_T_cam1_g, reuse=reuse)

        return total_loss, cam0_T_cam1_e, feat1_warped

    def multi_scale_corr3Dr(self, total_loss, feat0, feat1, vox_util, summ_writer, cam0_T_cam1_g=None, reuse=False, do_print=False):
        # the idea here is:
        # at each scale, find the answer, and then warp
        # to make the next scale closer to the answer
        # this allows a small displacement to be effective at each scale

        alignments = []

        B, C, Z, Y, X = list(feat0.size())
        utils.basic.assert_same_shape(feat0, feat1)

        summ_writer.summ_feat('ego/feat0', feat0, pca=True)
        summ_writer.summ_feat('ego/feat1', feat1, pca=True)

        if (cam0_T_cam1_g is not None):
            eye = utils.geom.eye_4x4(B)
            _ = eval_against_gt(0, eye, cam0_T_cam1_g, sc=0.0, summ_writer=summ_writer)

        feat1_backup = feat1.clone()

        rots = torch.linspace(-self.max_deg, self.max_deg, self.R)
        rots = torch.reshape(rots, [self.R])

        rot_cam_total = torch.zeros([B])
        delta_cam_total = torch.zeros([B, 3])

        for sc in self.scales:
            Z_ = int(Z*sc)
            Y_ = int(Y*sc)
            X_ = int(X*sc)

            if not sc==1.0:
                feat0_ = F.interpolate(feat0, scale_factor=sc, mode='trilinear')
                feat1_ = F.interpolate(feat1, scale_factor=sc, mode='trilinear')
            else:
                feat0_ = feat0.clone()
                feat1_ = feat1.clone()

            # have a heatmap at least sized 3, so that an argmax is capable of returning 0
            valid_Z = Z_-self.max_disp_z*2
            valid_Y = Y_-self.max_disp_y*2
            valid_X = X_-self.max_disp_x*2
            assert(valid_Z >= 3)
            assert(valid_Y >= 3)
            assert(valid_X >= 3)

            summ_writer.summ_feat('ego/feat0_resized_%.3f' % sc, feat0_, pca=True)
            summ_writer.summ_feat('ego/feat1_resized_%.3f' % sc, feat1_, pca=True)

            ## now we want to rotate the features into all of the orientations
            # first we define the orientations
            r0 = torch.zeros([B*self.R])
            ry = torch.unsqueeze(rots, axis=0).repeat([B, 1]).reshape([B*self.R])
            r = utils.geom.eul2rotm(r0, utils.geom.deg2rad(ry), r0)
            t = torch.zeros([B*self.R, 3])
            # this will carry us from "1" coords to "N" (new) coords
            camN_T_cam1 = utils.geom.merge_rt(r, t)
            # this is B*R x 4 x 4
            # we want to apply this to feat1
            # we first need the feats to lead with B*R
            feat0_ = torch.unsqueeze(feat0_, axis=1).repeat([1, self.R, 1, 1, 1, 1])
            feat1_ = torch.unsqueeze(feat1_, axis=1).repeat([1, self.R, 1, 1, 1, 1])
            feat0_ = feat0_.reshape([B*self.R, C, Z_, Y_, X_])
            feat1_ = feat1_.reshape([B*self.R, C, Z_, Y_, X_])

            featN_ = vox_util.apply_4x4_to_vox(camN_T_cam1, feat1_)

            featN__ = featN_.reshape([B, self.R, C, Z_, Y_, X_])
            summ_writer.summ_feats('ego/featN_%.3f_postwarp' % sc, torch.unbind(featN__, axis=1), pca=False)

            cc = cost_volume_3D(feat0_,
                                featN_,
                                max_disp_z=self.max_disp_z,
                                max_disp_y=self.max_disp_y,
                                max_disp_x=self.max_disp_x)

            # cc is B*R x Z_ x Y_ x X_ x E,
            # i.e., each spatial location has a heatmap squished into the E dim

            # reduce along the spatial dims
            heat = torch.sum(cc, axis=[2,3,4])
            # flesh out the heatmaps
            heat = heat.reshape([B, self.R, 1, self.E1, self.E2, self.E3])
            # have a look
            summ_writer.summ_oned('ego/heat_%.3f' % sc, torch.mean(heat[0], axis=-2, keepdim=False))

            feat = heat.reshape([B, self.R*self.E])
            feat = F.leaky_relu(feat, negative_slope=0.1)
            # relja said normalizing helps:
            feat_norm = utils.basic.l2_on_axis(feat, 1, keepdim=True)
            feat = feat/(EPS+feat_norm)
            feat = self.first_layer(feat)
            feat = F.leaky_relu(feat, negative_slope=0.1)
            feat = self.second_layer(feat)
            feat = F.leaky_relu(feat, negative_slope=0.1)
            feat = self.third_layer(feat)
            r, y, x, z = torch.unbind(feat, axis=1)

            # convert the mem argmax into a translation in cam coords
            xyz_argmax_mem = torch.unsqueeze(torch.stack([x, y, z], axis=1), axis=1)
            xyz_zero_mem = torch.zeros([B, 1, 3])
            # in the transformation, use Y*sc instead of Y_, in case we cropped instead of scaled
            xyz_argmax_cam = vox_util.Mem2Ref(xyz_argmax_mem.cuda(), int(Z*sc), int(Y*sc), int(X*sc))
            xyz_zero_cam = vox_util.Mem2Ref(xyz_zero_mem.cuda(), int(Z*sc), int(Y*sc), int(X*sc))
            xyz_delta_cam = xyz_argmax_cam-xyz_zero_cam

            # mem is aligned with cam, and scaling does not affect rotation
            rot_cam = r.clone()

            summ_writer.summ_histogram('xyz_delta_cam', xyz_delta_cam)
            summ_writer.summ_histogram('rot_cam', rot_cam)

            delta_cam_total += xyz_delta_cam.reshape([B, 3]).cpu()
            rot_cam_total += rot_cam.cpu()

            r0 = torch.zeros([B])
            cam0_T_cam1_e = utils.geom.merge_rt(utils.geom.eul2rotm(r0,
                                                                    utils.geom.deg2rad(rot_cam_total),
                                                                    r0),
                                                -delta_cam_total)
            # bring feat1_backup into alignment with feat0, using the cumulative RT
            # if the estimate were perfect, this would yield feat0, but let's continue to call it feat1
            feat1 = vox_util.apply_4x4_to_vox(cam0_T_cam1_e, feat1_backup)
            # we will use feat1 in the next iteration of the loop

            if (cam0_T_cam1_g is not None):
                total_loss = eval_against_gt(total_loss, cam0_T_cam1_e, cam0_T_cam1_g,
                                             t_coeff=hyp.ego_t_l2_coeff*sc,
                                             deg_coeff=hyp.ego_deg_l2_coeff*sc,
                                             sc=sc,
                                             summ_writer=summ_writer)

        return total_loss, cam0_T_cam1_e, feat1
