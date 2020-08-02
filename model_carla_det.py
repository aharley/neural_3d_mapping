import time
import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
import os

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.detnet import DetNet

import utils.vox
import utils.samp
import utils.geom
import utils.misc
import utils.improc
import utils.basic
# import utils.track
# import frozen_flow_net
import utils.eval

# from tensorboardX import SummaryWriter
# from backend import saverloader, inputs
# from torchvision import datasets, transforms

np.set_printoptions(precision=2)
np.random.seed(0)
# EPS = 1e-6
# MAX_QUEUE = 10 # how many items before the summaryWriter flushes

class CARLA_DET(Model):
    def initialize_model(self):
        print("------ INITIALIZING MODEL OBJECTS ------")
        self.model = CarlaDetModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        if hyp.do_freeze_det:
            self.model.detnet.eval()
            self.set_requires_grad(self.model.detnet, False)

class CarlaDetModel(nn.Module):
    def __init__(self):
        super(CarlaDetModel, self).__init__()
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_det:
            self.detnet = DetNet()
        
    def prepare_common_tensors(self, feed):
        results = dict()
        
        self.summ_writer = utils.improc.Summ_writer(
            writer=feed['writer'],
            global_step=feed['global_step'],
            log_freq=feed['set_log_freq'],
            fps=8,
            just_gif=True)
        global_step = feed['global_step']

        self.B = feed['set_batch_size']
        self.S = feed['set_seqlen']
        self.set_name = feed['set_name']

        # in det mode, we do not have much reason to have S>1
        assert(self.S==1)
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        
        self.N = hyp.N
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)

        self.pix_T_cams = feed['pix_T_cams']
        set_data_format = feed['set_data_format']
        self.S = feed['set_seqlen']

        self.origin_T_camRs = feed['origin_T_camRs']
        self.origin_T_camXs = feed['origin_T_camXs']

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camXs_T_camX0s = __u(utils.geom.safe_inverse(__p(self.camX0s_T_camXs)))
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed['xyz_camXs']
        
        if self.set_name=='test' or self.set_name=='val':
            scene_centroid_x = 0.0
            scene_centroid_y = 1.0
            scene_centroid_z = 18.0
            scene_centroid = np.array([scene_centroid_x,
                                       scene_centroid_y,
                                       scene_centroid_z]).reshape([1, 3])
            self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
            self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        else:
            # randomize a bit, as a form of data aug
            all_ok = False
            num_tries = 0
            while (not all_ok) and (num_tries < 100):
                scene_centroid_x = np.random.uniform(-8.0, 8.0)
                scene_centroid_y = np.random.uniform(-1.0, 3.0)
                scene_centroid_z = np.random.uniform(10.0, 26.0)
                scene_centroid = np.array([scene_centroid_x,
                                           scene_centroid_y,
                                           scene_centroid_z]).reshape([1, 3])
                self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
                num_tries += 1
                all_ok = True
                self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                # we want to ensure this gives us a few points inbound for each element
                inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camXs), self.Z, self.Y, self.X, already_mem=False))
                # this is B x S x N
                num_inb = torch.sum(inb.float(), axis=2)
                # this is B x S
                
                if torch.min(num_inb) < 300:
                    all_ok = False
            self.summ_writer.summ_scalar('centroid_sampling/num_tries', float(num_tries))
            self.summ_writer.summ_scalar('centroid_sampling/num_inb', torch.mean(num_inb).cpu().item())
            if num_tries >= 100:
                return False # not OK; do not train on this
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z

        origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
        boxlists = feed['boxlists']
        self.scorelist_s = feed['scorelists']
        self.tidlist_s = feed['tidlists']
        boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
        lrtlist_camRs_ = utils.misc.parse_boxes(boxlists_, origin_T_camRs_)
        self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        self.lrtlist_camR0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))

        inbound_s = __u(utils.misc.rescore_lrtlist_with_inbound(
            __p(self.lrtlist_camRs), __p(self.tidlist_s), self.Z, self.Y, self.X, self.vox_util))
        self.scorelist_s *= inbound_s

        for b in list(range(self.B)):
            if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
                return False # not OK; do not train on this
        
        self.rgb_camXs = feed['rgb_camXs']
        
        # get 3d voxelized inputs
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # these are B x C x Z x Y x X
        self.summ_writer.summ_occs('3d_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
        self.summ_writer.summ_unps('3d_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))

        return True # OK
        
    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        results = dict()

        # eliminate the seq dim, to make life easier
        lrtlist_camX = self.lrtlist_camXs[:, 0]
        rgb_camX0 = self.rgb_camXs[:,0]
        occ_memX0 = self.occ_memXs[:,0]
        unp_memX0 = self.unp_memXs[:,0]
        tidlist_g = self.tidlist_s[:,0]
        scorelist_g = self.scorelist_s[:,0]

        if hyp.do_feat3d:
            # start with a 4-channel feature map;
            feat_memX0_input = torch.cat([
                occ_memX0,
                unp_memX0*occ_memX0,
            ], dim=1)

            # featurize
            feat3d_loss, feat_halfmemX0 = self.feat3dnet(
                feat_memX0_input,
                self.summ_writer)
            total_loss += feat3d_loss

            self.summ_writer.summ_feat('feat3d/feat_memX0_input', feat_memX0_input, pca=True)
            self.summ_writer.summ_feat('feat3d/feat_halfmemX0', feat_halfmemX0, pca=True)
            
        if hyp.do_det:

            # this detector can only handle axis-aligned boxes (like rcnn)
            axlrtlist_camX = utils.geom.inflate_to_axis_aligned_lrtlist(lrtlist_camX)
            lrtlist_memX = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_camX, self.Z, self.Y, self.X)
            axlrtlist_memX = utils.geom.inflate_to_axis_aligned_lrtlist(lrtlist_memX)
            self.summ_writer.summ_lrtlist_bev(
                'det/boxlist_g',
                occ_memX0[0:1],
                lrtlist_memX[0:1],
                scorelist_g,
                tidlist_g, 
                self.vox_util, 
                already_mem=True)
            self.summ_writer.summ_lrtlist_bev(
                'det/axboxlist_g',
                occ_memX0[0:1],
                axlrtlist_memX[0:1],
                scorelist_g,
                tidlist_g, 
                self.vox_util, 
                already_mem=True)
            
            lrtlist_halfmemX = self.vox_util.apply_mem_T_ref_to_lrtlist(lrtlist_camX, self.Z2, self.Y2, self.X2)
            axlrtlist_halfmemX = utils.geom.inflate_to_axis_aligned_lrtlist(lrtlist_halfmemX)

            det_loss, boxlist_halfmemX_e, scorelist_e, tidlist_e, pred_objectness, sco, ove = self.detnet(
                axlrtlist_halfmemX,
                scorelist_g,
                feat_halfmemX0,
                self.summ_writer)
            total_loss += det_loss
            
            lrtlist_halfmemX_e = utils.geom.convert_boxlist_to_lrtlist(boxlist_halfmemX_e)
            lrtlist_camX_e = self.vox_util.apply_ref_T_mem_to_lrtlist(lrtlist_halfmemX_e, self.Z2, self.Y2, self.X2)

            lrtlist_e = lrtlist_camX_e[0:1]
            lrtlist_g = lrtlist_camX[0:1]
            scorelist_e = scorelist_e[0:1]
            scorelist_g = scorelist_g[0:1]
            lrtlist_e, lrtlist_g, scorelist_e, scorelist_g = utils.eval.drop_invalid_lrts(
                lrtlist_e, lrtlist_g, scorelist_e, scorelist_g)

            lenlist_e, _ = utils.geom.split_lrtlist(lrtlist_e)
            clist_e = utils.geom.get_clist_from_lrtlist(lrtlist_e)
            lenlist_g, _ = utils.geom.split_lrtlist(lrtlist_g)
            clist_g = utils.geom.get_clist_from_lrtlist(lrtlist_g)
            axlenlist_g, _ = utils.geom.split_lrtlist(axlrtlist_camX[0:1])
            axclist_g = utils.geom.get_clist_from_lrtlist(axlrtlist_camX[0:1])

            _, Ne, _ = list(lrtlist_e.shape)
            _, Ng, _ = list(lrtlist_g.shape)
            # only summ if there is at least one pred and one gt
            if Ne > 0 and Ng > 0:
                lrtlist_e_ = lrtlist_e.unsqueeze(2).repeat(1, 1, Ng, 1).reshape(1, Ne * Ng, -1)
                lrtlist_g_ = lrtlist_g.unsqueeze(1).repeat(1, Ne, 1, 1).reshape(1, Ne * Ng, -1)
                ious, _ = utils.geom.get_iou_from_corresponded_lrtlists(lrtlist_e_, lrtlist_g_)
                ious = ious.reshape(1, Ne, Ng)
                ious_e = torch.max(ious, dim=2)[0]
                self.summ_writer.summ_lrtlist(
                    'det/boxlist_eg',
                    rgb_camX0[0:1],
                    torch.cat((lrtlist_g, lrtlist_e), dim=1),
                    torch.cat((ious_e.new_ones(1, Ng), ious_e), dim=1),
                    torch.cat([torch.ones(1, Ng).long().cuda(),
                               torch.ones(1, Ne).long().cuda()+1], dim=1),
                    self.pix_T_cams[0:1, 0])
                self.summ_writer.summ_lrtlist_bev(
                    'det/boxlist_bev_eg',
                    occ_memX0[0:1],
                    torch.cat((lrtlist_g, lrtlist_e), dim=1),
                    torch.cat((ious_e.new_ones(1, Ng), ious_e), dim=1),
                    torch.cat([torch.ones(1, Ng).long().cuda(),
                               torch.ones(1, Ne).long().cuda()+1], dim=1),
                    self.vox_util, 
                    already_mem=False)

            ious = [0.3, 0.4, 0.5, 0.6, 0.7]
            maps_3d, maps_2d = utils.eval.get_mAP_from_lrtlist(lrtlist_e, scorelist_e, lrtlist_g, ious)
            for ind, overlap in enumerate(ious):
                self.summ_writer.summ_scalar('ap_3d/%.2f_iou' % overlap, maps_3d[ind])
                self.summ_writer.summ_scalar('ap_bev/%.2f_iou' % overlap, maps_2d[ind])

            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False
        
        
    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)

        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            else:
                print('not prepared for this set_name:', set_name)
                assert(False)

    
