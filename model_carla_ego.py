import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
# import imageio,scipy

from model_base import Model
from nets.occnet import OccNet
from nets.feat2dnet import Feat2dNet
from nets.feat3dnet import Feat3dNet
from nets.emb2dnet import Emb2dNet
from nets.emb3dnet import Emb3dNet
from nets.viewnet import ViewNet
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
        if hyp.do_freeze_view:
            self.model.viewnet.eval()
            self.set_requires_grad(self.model.viewnet, False)
        if hyp.do_freeze_occ:
            self.model.occnet.eval()
            self.set_requires_grad(self.model.occnet, False)
        if hyp.do_freeze_emb2d:
            self.model.emb2dnet.eval()
            self.set_requires_grad(self.model.emb2dnet, False)

        if hyp.do_emb2d:
            # freeze the slow model
            self.model.feat2dnet_slow.eval()
            self.set_requires_grad(self.model.feat2dnet_slow, False)
        if hyp.do_emb3d:
            # freeze the slow model
            self.model.feat3dnet_slow.eval()
            self.set_requires_grad(self.model.feat3dnet_slow, False)
            
class CarlaEgoModel(nn.Module):
    def __init__(self):
        super(CarlaEgoModel, self).__init__()
        if hyp.do_occ:
            self.occnet = OccNet()
        if hyp.do_view:
            self.viewnet = ViewNet()

        if hyp.do_feat2d:
            self.feat2dnet = Feat2dNet()
        if hyp.do_emb2d:
            self.emb2dnet = Emb2dNet()
            # make a slow net
            self.feat2dnet_slow = Feat2dNet(in_dim=3)
            
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=1)
        if hyp.do_ego:
            self.egonet = EgoNet(
                R=hyp.ego_num_rots,
                rot_max=hyp.ego_rot_max,
                num_scales=hyp.ego_num_scales,
                max_disp_h=hyp.ego_max_disp_h,
                max_disp_w=hyp.ego_max_disp_w,
                max_disp_d=hyp.ego_max_disp_d)

        if hyp.do_emb3d:
            self.emb3dnet = Emb3dNet()
            # make a slow net
            self.feat3dnet_slow = Feat3dNet(in_dim=1)

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

        # if self.set_name=='test':
        #     self.Z, self.Y, self.X = hyp.Z_test, hyp.Y_test, hyp.X_test
        # elif self.set_name=='val':
        #     self.Z, self.Y, self.X = hyp.Z_val, hyp.Y_val, hyp.X_val
        # else:
        self.Z, self.Y, self.X = hyp.Z, hyp.Y, hyp.X
        self.Z2, self.Y2, self.X2 = int(self.Z/2), int(self.Y/2), int(self.X/2)
        self.Z4, self.Y4, self.X4 = int(self.Z/4), int(self.Y/4), int(self.X/4)

        self.ZZ, self.ZY, self.ZX = hyp.ZZ, hyp.ZY, hyp.ZX
        self.pix_T_cams = feed['pix_T_cams']
        set_data_format = feed['set_data_format']
        self.S = feed['set_seqlen']
        

        self.origin_T_cams = feed['origin_T_camXs']
        # self.camX0s_T_cams = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)

        self.xyz_cams = feed['xyz_camXs']
        # self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0
        scene_centroid_z = 18.0
        scene_centroid = np.array([scene_centroid_x,
                                   scene_centroid_y,
                                   scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        # if self.set_name=='test' or self.set_name=='val':
        #     # fixed centroid
        #     scene_centroid_x = 0.0
        #     scene_centroid_y = 1.0
        #     scene_centroid_z = 18.0
        # else:
        #     # randomize a bit, as a form of data aug
        #     all_ok = False
        #     num_tries = 0
        #     while (not all_ok) and (num_tries < 100):
        #         scene_centroid_x = np.random.uniform(-8.0, 8.0)
        #         scene_centroid_y = np.random.uniform(-1.5, 3.0)
        #         scene_centroid_z = np.random.uniform(10.0, 26.0)
        #         scene_centroid = np.array([scene_centroid_x,
        #                                    scene_centroid_y,
        #                                    scene_centroid_z]).reshape([1, 3])
        #         self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
        #         num_tries += 1
        #         all_ok = True
        #         self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
        #         # we want to ensure this gives us a few points inbound for each element
        #         inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z, self.Y, self.X, already_mem=False))
        #         # this is B x S x N
        #         num_inb = torch.sum(inb.float(), axis=2)
        #         # this is B x S
                
        #         if torch.min(num_inb) < 300:
        #             all_ok = False
        #     self.summ_writer.summ_scalar('centroid_sampling/num_tries', float(num_tries))
        #     self.summ_writer.summ_scalar('centroid_sampling/num_inb', torch.mean(num_inb).cpu().item())
        #     if num_tries >= 100:
        #         return False
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        self.rgb_camXs = feed['rgb_camXs']

        # get 3d voxelized inputs
        self.occ_mems = __u(self.vox_util.voxelize_xyz(__p(self.xyz_cams), self.Z, self.Y, self.X))
        # self.unp_mems = __u(self.vox_util.unproject_rgb_to_mem(
        #     __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # these are B x C x Z x Y x X
        self.summ_writer.summ_occs('3d_inputs/occ_mems', torch.unbind(self.occ_mems, dim=1))
        # self.summ_writer.summ_unps('3d_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
        
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
        # cam1_T_cam0 = utils.geom.safe_inverse(cam0_T_cam1)

        #     occ_1_mem = self.vox_util.voxelize_xyz(xyz_cam1, self.Z, self.Y, self.X)
        feat_mems_input = self.occ_mems.clone()
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

        # try aligning the frames
        occ_mem0_e = self.vox_util.apply_4x4_to_vox(cam0_T_cam1_e, self.occ_mems[:,1])
        self.summ_writer.summ_occs('ego/occs_aligned', [occ_mem0_e, self.occ_mems[:,0]])
        self.summ_writer.summ_occs('ego/occs_unaligned', [self.occ_mems[:,0], self.occ_mems[:,1]])
        
        # results['poses'] = []

        # # occ_mems is B x C x 1 x Z x Y x X
        # occ_mems = []

        # feats_1_halfmem = None
        # occ_1_mem = None

        # # We might be able to allow for a larger displacement based on pyramid effects,
        # # but the math starts to get complex and this should be good enough
        # distance_bounds = torch.Tensor([hyp.ZMAX - hyp.ZMIN,
        #                                 hyp.XMAX - hyp.XMIN])
        # spatial_feat_dims = torch.tensor([hyp.Z, hyp.X])
        # # In meters
        # voxel_dims = distance_bounds / spatial_feat_dims / (1./hyp.ego_num_scales)
        # max_allowable_displacement = torch.min(voxel_dims * torch.Tensor([hyp.ego_max_disp_d, hyp.ego_max_disp_w]))

        # # We wish to iterate over the sequence dimension across all batches
        # for index in range(self.S):
        #     if self.S > 2:
        #         print(f'Handling iter {index}')

        #     feats_0_halfmem = feats_1_halfmem
        #     occ_0_mem = occ_1_mem

        #     xyz_cam1 = self.xyz_cams[:, index]

        #     if index != 0:
        #         # Build the ground truth matrix so we can calculate error
        #         origin_T_cam1 = self.origin_T_cams[:, index]
        #         origin_T_cam0 = self.origin_T_cams[:, index-1]
        #         cam0_T_cam1 = utils.basic.matmul2(utils.geom.safe_inverse(origin_T_cam0), origin_T_cam1)
        #         cam1_T_cam0 = utils.geom.safe_inverse(cam0_T_cam1)

        #         # # Handle synthetic transformations if necessary
        #         # if feed['set_name'] == 'train' and torch.randn(1) < hyp.ego_synth_prob:
        #         #     synthcam_T_cam0 = utils.geom.get_random_rt(self.B,
        #         #                                                r_amount=hyp.ego_rot_max,
        #         #                                                t_amount=max_allowable_displacement)
        #         #     synthcam_T_cam1 = utils.basic.matmul2(synthcam_T_cam0, cam0_T_cam1)
        #         #     xyz_synthcam = utils.geom.apply_4x4(synthcam_T_cam1, xyz_cam1)

        #         #     xyz_cam1 = xyz_synthcam
        #         #     cam1_T_cam0 = synthcam_T_cam0
        #         #     cam0_T_cam1 = utils.geom.safe_inverse(cam1_T_cam0)
        #         #     origin_T_cam1 = utils.basic.matmul2(origin_T_cam0, cam0_T_cam1)

        #     occ_1_mem = self.vox_util.voxelize_xyz(xyz_cam1, self.Z, self.Y, self.X)
        #     # unp_1_mem = utils.vox.unproject_rgb_to_mem(self.rgb_cams[:, index], self.Z, self.Y, self.X, self.pix_T_cams[:, index])

        #     occ_mems.append(occ_1_mem.clone().cpu())
        #     # unp_mems.append(unp_1_mem.clone().cpu())
        #     feats_input = occ_1_mem.clone()

        #     feat_loss, feats_1_halfmem = self.feat3dnet(feats_input, self.summ_writer)
        #     total_loss += feat_loss

        #     if index == 0:
        #         continue

        #     # Estimated transformation is previous_T_current
        #     transformation_loss, cam0_T_cam1_e, _ = self.egonet(
        #         feats_0_halfmem,
        #         feats_1_halfmem,
        #         cam0_T_cam1,
        #         self.vox_util,
        #         self.summ_writer)
        #     total_loss += transformation_loss

        #     cam1_T_cam0_e = utils.geom.safe_inverse(cam0_T_cam1_e.cpu())

        #     # Save the estimated transformation so that we can visualize it later
        #     if len(results['poses']) == 0:
        #         results['poses'].append(origin_T_cam0)

        #     combined = utils.basic.matmul2(results['poses'][-1], cam0_T_cam1_e)
        #     results['poses'].append(combined)


        #     # Transform the second frame's occupancy based on our estimated transformation
        #     transformed_occs = self.vox_util.apply_4x4_to_vox(cam0_T_cam1_e.cuda(), occ_1_mem)
        #     # Visualize this result and compare it to the actual first frame
        #     self.summ_writer.summ_occs('3d_outputs/transformed_occ', (transformed_occs, occ_0_mem))
        #     # Visualize the untrasformed frames so we can see if we're improving
        #     self.summ_writer.summ_occs('3d_outputs/occs_original', (occ_1_mem, occ_0_mem))

        # # self.summ_writer.summ_feats('3d_feats/feats_input', torch.unbind(feats_input, dim=0), pca=False)
        # self.summ_writer.summ_scalar('transformation_loss/mse', transformation_loss)

        # # visualize what we got
        # # if self.include_image_summs:
        # self.summ_writer.summ_occs('3d_inputs/occ_mems', [occ.cuda() for occ in occ_mems])#, use_cuda=False)
        # # self.summ_writer.summ_unps('3d_inputs/unp_mems', unp_mems, occ_mems)
        # # rgb_cams_cpu = [rgb_cam.cpu() for rgb_cam in self.rgb_cams]
        # # self.summ_writer.summ_rgbs('2D_inputs/rgb_cams', rgb_cams_cpu)

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
            # elif self.set_name=='test':
            #     return self.run_test(feed)
            else:
                print('weird set_name:', set_name)
                assert(False)
