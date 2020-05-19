import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
# import imageio,scipy

from model_base import Model
from nets.feat3dnet import Feat3dNet
from nets.egonet import EgoNet

import torch.nn.functional as F

import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_EGO(Model):
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaEgoModel()
        if hyp.do_freeze_feat3d:
            self.model.feat3dnet.eval()
            self.set_requires_grad(self.model.feat3dnet, False)
        if hyp.do_freeze_ego:
            self.model.egonet.eval()
            self.set_requires_grad(self.model.egonet, False)
            
class CarlaEgoModel(nn.Module):
    def __init__(self):
        super(CarlaEgoModel, self).__init__()
        
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_ego:
            self.egonet = EgoNet(
                num_scales=hyp.ego_num_scales,
                num_rots=hyp.ego_num_rots,
                max_deg=hyp.ego_max_deg,
                max_disp_z=hyp.ego_max_disp_z,
                max_disp_y=hyp.ego_max_disp_y,
                max_disp_x=hyp.ego_max_disp_x)

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
        
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        self.H, self.W, self.V, self.N = hyp.H, hyp.W, hyp.V, hyp.N
        self.PH, self.PW = hyp.PH, hyp.PW

        if self.set_name=='test':
            self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        elif self.set_name=='val':
            self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        else:
            self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed['pix_T_cams']
        self.S = feed['set_seqlen']

        # in this mode, we never use R coords, so we can drop the R/X notation
        self.origin_T_cams = feed['origin_T_camXs']
        self.xyz_cams = feed['xyz_camXs']

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_cams = feed['rgb_camXs']

        # get 3d voxelized inputs
        self.occ_mems = __u(self.vox_util.voxelize_xyz(__p(self.xyz_cams), self.Z, self.Y, self.X))
        self.unp_mems = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_cams), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # these are B x C x Z x Y x X
        self.summ_writer.summ_occs('3d_inputs/occ_mems', torch.unbind(self.occ_mems, dim=1))
        self.summ_writer.summ_unps('3d_inputs/unp_mems', torch.unbind(self.unp_mems, dim=1), torch.unbind(self.occ_mems, dim=1))
        
        return True # OK

    def run_train(self, feed):
        total_loss = torch.tensor(0.0).cuda()
        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)
        results = dict()

        assert(hyp.do_ego)
        assert(self.S==2)

        origin_T_cam0 = self.origin_T_cams[:, 0]
        origin_T_cam1 = self.origin_T_cams[:, 1]
        cam0_T_cam1 = utils.basic.matmul2(utils.geom.safe_inverse(origin_T_cam0), origin_T_cam1)

        feat_mems_input = torch.cat([
            self.occ_mems,
            self.occ_mems*self.unp_mems,
        ], dim=2)
        feat_loss, feat_halfmems_ = self.feat3dnet(__p(feat_mems_input), self.summ_writer)
        feat_halfmems = __u(feat_halfmems_)
        total_loss += feat_loss
        
        ego_loss, cam0_T_cam1_e, _ = self.egonet(
            feat_halfmems[:,0],
            feat_halfmems[:,1],
            cam0_T_cam1,
            self.vox_util,
            self.summ_writer)
        total_loss += ego_loss

        # try aligning the frames, for a qualitative result
        occ_mem0_e = self.vox_util.apply_4x4_to_vox(cam0_T_cam1_e, self.occ_mems[:,1])
        self.summ_writer.summ_occs('ego/occs_aligned', [occ_mem0_e, self.occ_mems[:,0]])
        self.summ_writer.summ_occs('ego/occs_unaligned', [self.occ_mems[:,0], self.occ_mems[:,1]])

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
