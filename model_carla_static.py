import torch
import torch.nn as nn
import hyperparams as hyp
import numpy as np
# import imageio,scipy
# from sklearn.cluster import KMeans

from model_base import Model
from nets.occnet import OccNet
from nets.feat2dnet import Feat2dNet
from nets.feat3dnet import Feat3dNet
from nets.emb2dnet import Emb2dNet
from nets.emb3dnet import Emb3dNet
from nets.viewnet import ViewNet

import torch.nn.functional as F

# from utils.basic import *
import utils.vox
import utils.samp
import utils.geom
import utils.improc
import utils.basic
import utils.eval
import utils.misc

np.set_printoptions(precision=2)
np.random.seed(0)

class CARLA_STATIC(Model):
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    def initialize_model(self):
        print('------ INITIALIZING MODEL OBJECTS ------')
        self.model = CarlaStaticModel()
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
            
class CarlaStaticModel(nn.Module):
    def __init__(self):
        super(CarlaStaticModel, self).__init__()
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
            # init slow params with fast params
            self.feat2dnet_slow.load_state_dict(self.feat2dnet.state_dict())
            
        if hyp.do_feat3d:
            self.feat3dnet = Feat3dNet(in_dim=4)
        if hyp.do_emb3d:
            self.emb3dnet = Emb3dNet()
            # make a slow net
            self.feat3dnet_slow = Feat3dNet(in_dim=4)
            # init slow params with fast params
            self.feat3dnet_slow.load_state_dict(self.feat3dnet.state_dict())

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
        

        self.origin_T_camRs = feed['origin_T_camRs']
        self.origin_T_camXs = feed['origin_T_camXs']

        self.camX0s_T_camXs = utils.geom.get_camM_T_camXs(self.origin_T_camXs, ind=0)
        self.camR0s_T_camRs = utils.geom.get_camM_T_camXs(self.origin_T_camRs, ind=0)
        self.camRs_T_camR0s = __u(utils.geom.safe_inverse(__p(self.camR0s_T_camRs)))
        self.camRs_T_camXs = __u(torch.matmul(__p(self.origin_T_camRs).inverse(), __p(self.origin_T_camXs)))
        self.camXs_T_camRs = __u(__p(self.camRs_T_camXs).inverse())

        self.xyz_camXs = feed['xyz_camXs']
        self.xyz_camRs = __u(utils.geom.apply_4x4(__p(self.camRs_T_camXs), __p(self.xyz_camXs)))
        self.xyz_camX0s = __u(utils.geom.apply_4x4(__p(self.camX0s_T_camXs), __p(self.xyz_camXs)))
        
        if self.set_name=='test' or self.set_name=='val':
            # fixed centroid
            scene_centroid_x = 0.0
            scene_centroid_y = 1.0
            scene_centroid_z = 18.0
        else:
            # randomize a bit, as a form of data aug
            all_ok = False
            num_tries = 0
            while (not all_ok) and (num_tries < 100):
                scene_centroid_x = np.random.uniform(-8.0, 8.0)
                scene_centroid_y = np.random.uniform(-1.5, 3.0)
                scene_centroid_z = np.random.uniform(10.0, 26.0)
                scene_centroid = np.array([scene_centroid_x,
                                           scene_centroid_y,
                                           scene_centroid_z]).reshape([1, 3])
                self.scene_centroid = torch.from_numpy(scene_centroid).float().cuda()
                num_tries += 1
                all_ok = True
                self.vox_util = utils.vox.Vox_util(self.Z, self.Y, self.X, self.set_name, scene_centroid=self.scene_centroid, assert_cube=True)
                # we want to ensure this gives us a few points inbound for each element
                inb = __u(self.vox_util.get_inbounds(__p(self.xyz_camX0s), self.Z, self.Y, self.X, already_mem=False))
                # this is B x S x N
                num_inb = torch.sum(inb.float(), axis=2)
                # this is B x S
                
                if torch.min(num_inb) < 300:
                    all_ok = False
            self.summ_writer.summ_scalar('centroid_sampling/num_tries', float(num_tries))
            self.summ_writer.summ_scalar('centroid_sampling/num_inb', torch.mean(num_inb).cpu().item())
            if num_tries >= 100:
                return False
        
        self.vox_size_X = self.vox_util.default_vox_size_X
        self.vox_size_Y = self.vox_util.default_vox_size_Y
        self.vox_size_Z = self.vox_util.default_vox_size_Z
        
        # _boxlist_camRs = feed['boxlists']
        # _tidlist_s = feed['tidlists'] # coordinate-less and plural
        # _scorelist_s = feed['scorelists'] # coordinate-less and plural
        # _scorelist_s = __u(utils.misc.rescore_boxlist_with_inbound(
        #     utils.geom.eye_4x4(self.B*self.S),
        #     __p(_boxlist_camRs),
        #     __p(_tidlist_s),
        #     self.Z, self.Y, self.X,
        #     self.vox_util,
        #     only_cars=False, pad=2.0))
        # boxlist_camRs_, tidlist_s_, scorelist_s_ = utils.misc.shuffle_valid_and_sink_invalid_boxes(
        #     __p(_boxlist_camRs), __p(_tidlist_s), __p(_scorelist_s))
        # self.boxlist_camRs = __u(boxlist_camRs_)
        # self.tidlist_s = __u(tidlist_s_)
        # self.scorelist_s = __u(scorelist_s_)

        # for b in list(range(self.B)):
        #     # if torch.sum(scorelist_s[b,0]) == 0:
        #     if torch.sum(self.scorelist_s[:,0]) < (self.B/2): # not worth it; return early
        #         return 0.0, None, True

        # lrtlist_camRs_, obj_lens_ = utils.misc.parse_boxes(__p(feed['boxlists']), __p(self.origin_T_camRs))
        origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4).repeat(1, 1, self.N, 1, 1).reshape(self.B*self.S, self.N, 4, 4)
        boxlists = feed['boxlists']
        self.scorelist_s = feed['scorelists']
        self.tidlist_s = feed['tidlists']
        # print('boxlists', boxlists.shape)
        boxlists_ = boxlists.reshape(self.B*self.S, self.N, 9)
        lrtlist_camRs_, _ = utils.misc.parse_boxes(boxlists_, origin_T_camRs_)
        self.lrtlist_camRs = lrtlist_camRs_.reshape(self.B, self.S, self.N, 19)
        
        # origin_T_camRs_ = self.origin_T_camRs.reshape(self.B, self.S, 1, 4, 4)
        # self.lrtlist_camRs = utils.misc.parse_boxes(box_camRs, origin_T_camRs)
        # self.lrtlist_camRs = __u(utils.geom.convert_boxlist_to_lrtlist(__p(self.boxlist_camRs)))
        self.lrtlist_camR0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camR0s_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(self.lrtlist_camRs)))
        self.lrtlist_camX0s = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camX0s_T_camXs), __p(self.lrtlist_camXs)))

        # self.crop_guess = (40,40,40)
        # self.crop_guess = (20,20,20)
        # self.crop_guess = (52,52,52)
        # self.crop_guess = (50,50,50)
        # self.crop_guess = (1,1,1)
        # self.crop_guess = (2,2,2)
        self.crop_guess = (4,4,4)
        # self.crop_guess = (25,25,25)
        if hyp.do_center:
            lrtlist = self.lrtlist_camX0s[:,0]
            clist = utils.geom.get_clist_from_lrtlist(lrtlist)
            # this is B x N x 3
            mask = self.vox_util.xyz2circles(clist, self.Z, self.Y, self.X, radius=1.0, soft=True, already_mem=False)
            mask = mask[:,:,
                        self.crop_guess[0]:-self.crop_guess[0],
                        self.crop_guess[1]:-self.crop_guess[2],
                        self.crop_guess[2]:-self.crop_guess[2]]
            self.center_mask = torch.max(mask, dim=1, keepdim=True)[0]

            mask_max = torch.max(self.center_mask.reshape(self.B, -1), dim=1)[0]
            # print('mask_max', mask_max.detach().cpu().numpy())
            # if torch.min(mask_max) < 1.0:
            #     # print('returning early!!!')
            #     # at least one ex has no objects in the crop; let's return early
            #     return False
        
        # self.summ_writer.summ_rgb('2d_inputs/rgb_camX0', self.rgb_camXs[:,0])
        # # self.occ_memX0s = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camX0s), self.Z, self.Y, self.X))
        # # self.summ_writer.summ_rgbs('2d_inputs/rgb_camXs', torch.unbind(self.rgb_camXs, dim=1))
        # # self.summ_writer.summ_occs('3d_inputs/occ_memRs', torch.unbind(self.occ_memRs, dim=1))
        # # self.summ_writer.summ_occs('3d_inputs/occ_memR0s', torch.unbind(self.occ_memR0s, dim=1))
        # # self.summ_writer.summ_occs('3d_inputs/occ_memX0s', torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_unps('3d_inputs/unp_memX0s', torch.unbind(self.unp_memX0s, dim=1), torch.unbind(self.occ_memX0s, dim=1))
        # # self.summ_writer.summ_occs('3d_inputs/obj_occR0s', torch.unbind(self.obj_occR0s, dim=1))
        # # self.summ_writer.summ_feat('3d_inputs/obj_mask', self.obj_mask_template, pca=False)


        self.rgb_camXs = feed['rgb_camXs']
        visX_e = []
        for s in list(range(0, self.S, 2)):
            visX_e.append(self.summ_writer.summ_lrtlist(
                '', self.rgb_camXs[:,s],
                self.lrtlist_camXs[:,s],
                self.scorelist_s[:,s],
                self.tidlist_s[:,s],
                self.pix_T_cams[:,s], only_return=True))
        self.summ_writer.summ_rgbs('obj/box_camXs_g', visX_e)

        ## get the projected depthmap and inbound mask
        self.depth_camXs_, self.valid_camXs_ = utils.geom.create_depth_image(__p(self.pix_T_cams), __p(self.xyz_camXs), self.H, self.W)
        self.dense_xyz_camXs_ = utils.geom.depth2pointcloud(self.depth_camXs_, __p(self.pix_T_cams))
        # we need to go to X0 to see what will be inbounds
        self.dense_xyz_camX0s_ = utils.geom.apply_4x4(__p(self.camX0s_T_camXs), self.dense_xyz_camXs_)
        self.inbound_camXs_ = self.vox_util.get_inbounds(self.dense_xyz_camX0s_, self.Z, self.Y, self.X).float()
        self.inbound_camXs_ = torch.reshape(self.inbound_camXs_, [self.B*self.S, 1, self.H, self.W])
        self.depth_camXs = __u(self.depth_camXs_)
        self.valid_camXs = __u(self.valid_camXs_) * __u(self.inbound_camXs_)
        self.summ_writer.summ_oned('2d_inputs/depth_camX0', self.depth_camXs[:,0], maxval=32.0)
        self.summ_writer.summ_oned('2d_inputs/valid_camX0', self.valid_camXs[:,0], norm=False)
        self.summ_writer.summ_rgb('2d_inputs/rgb_camX0', self.rgb_camXs[:,0])

        # get 3d voxelized inputs
        self.occ_memXs = __u(self.vox_util.voxelize_xyz(__p(self.xyz_camXs), self.Z, self.Y, self.X))
        self.unp_memXs = __u(self.vox_util.unproject_rgb_to_mem(
            __p(self.rgb_camXs), self.Z, self.Y, self.X, __p(self.pix_T_cams)))
        # these are B x C x Z x Y x X
        self.summ_writer.summ_occs('3d_inputs/occ_memXs', torch.unbind(self.occ_memXs, dim=1))
        self.summ_writer.summ_unps('3d_inputs/unp_memXs', torch.unbind(self.unp_memXs, dim=1), torch.unbind(self.occ_memXs, dim=1))
        
        return True # OK

    def run_train(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        #####################
        ## run the nets
        #####################

        if hyp.do_feat2d:
            feat2d_loss, feat_camX0 = self.feat2dnet(
                self.rgb_camXs[:,0],
                self.summ_writer,
            )
            if hyp.do_emb2d:
                # for stability, we will also use a slow net here
                _, altfeat_camX0 = self.feat2dnet_slow(self.rgb_camXs[:,0])
        
        if hyp.do_feat3d:
            # start with a 4-channel feature map;
            feat_memXs_input = torch.cat([
                self.occ_memXs,
                self.unp_memXs*self.occ_memXs,
            ], dim=2)

            # featurize
            feat3d_loss, feat_memXs_ = self.feat3dnet(
                __p(feat_memXs_input[:,1:]), self.summ_writer)
            feat_memXs = __u(feat_memXs_)
            total_loss += feat3d_loss

            valid_memXs = torch.ones_like(feat_memXs[:,:,0:1])
            feat_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:,1:], feat_memXs)
            valid_memRs = self.vox_util.apply_4x4s_to_voxs(self.camRs_T_camXs[:,1:], valid_memXs)
            # these are B x S x C x Z2 x Y2 x X2

            feat_memR = utils.basic.reduce_masked_mean(
                feat_memRs, valid_memRs, dim=1)
            valid_memR = torch.max(valid_memRs, dim=1)[0]
            # these are B x C x Z2 x Y2 x X2
            self.summ_writer.summ_feat('feat3d/feat_output_agg', feat_memR, valid_memR, pca=True)

            if hyp.do_emb3d:
                _, altfeat_memR = self.feat3dnet_slow(feat_memXs_input[:,0])
                altvalid_memR = torch.ones_like(altfeat_memR[:,0:1])
                self.summ_writer.summ_feat('feat3d/altfeat_input', feat_memXs_input[:,0], pca=True)
                self.summ_writer.summ_feat('feat3d/altfeat_output', altfeat_memR, pca=True)
            
        if hyp.do_occ:
            assert(hyp.do_feat3d)
            occ_memR_sup, free_memR_sup, _, _ = self.vox_util.prep_occs_supervision(
                self.camRs_T_camXs,
                self.xyz_camXs,
                self.Z2, self.Y2, self.X2,
                agg=True)
            occ_loss, occ_memR_pred = self.occnet(
                feat_memR, 
                occ_memR_sup,
                free_memR_sup,
                valid_memR, 
                self.summ_writer)
            total_loss += occ_loss

        if hyp.do_view:
            assert(hyp.do_feat3d)
            # decode the perspective volume into an image
            view_loss, rgb_camX0_e, viewfeat_camX0 = self.viewnet(
                self.pix_T_cams[:,0],
                self.camXs_T_camRs[:,0],
                feat_memR, 
                self.rgb_camXs[:,0],
                self.vox_util,
                valid=self.valid_camXs[:,0],
                summ_writer=self.summ_writer)
            total_loss += view_loss
            
        if hyp.do_emb2d:
            assert(hyp.do_feat2d)
            
            if hyp.do_view:
                # anchor against the bottom-up 2d net
                valid_camX0 = F.interpolate(self.valid_camXs[:,0], scale_factor=0.5, mode='nearest')
                emb2d_loss, _ = self.emb2dnet(
                    viewfeat_camX0,
                    feat_camX0,
                    valid_camX0,
                    summ_writer=self.summ_writer,
                    suffix='_view')
                total_loss += emb2d_loss

            # anchor against the slow net
            emb2d_loss, _ = self.emb2dnet(
                feat_camX0,
                altfeat_camX0,
                torch.ones_like(feat_camX0[:,0:1]),
                summ_writer=self.summ_writer,
                suffix='_slow')
            total_loss += emb2d_loss

        if hyp.do_emb3d:
            assert(hyp.do_feat3d)
            # compute 3D ML
            emb3d_loss = self.emb3dnet(
                feat_memR,
                altfeat_memR,
                valid_memR.round(),
                altvalid_memR.round(),
                self.summ_writer)
            total_loss += emb3d_loss
            
        self.summ_writer.summ_scalar('loss', total_loss.cpu().item())
        return total_loss, results, False

    def run_test(self, feed):
        results = dict()

        global_step = feed['global_step']
        total_loss = torch.tensor(0.0).cuda()
        # total_loss = torch.autograd.Variable(0.0, requires_grad=True).cuda()

        __p = lambda x: utils.basic.pack_seqdim(x, self.B)
        __u = lambda x: utils.basic.unpack_seqdim(x, self.B)

        # get the boxes
        boxlist_camRs = feed['boxlists']
        tidlist_s = feed['tidlists'] # coordinate-less and plural
        scorelist_s = feed['scorelists'] # coordinate-less and plural
        
        lrtlist_camRs = __u(utils.geom.convert_boxlist_to_lrtlist(__p(boxlist_camRs))).reshape(self.B, self.S, self.N, 19)
        lrtlist_camXs = __u(utils.geom.apply_4x4_to_lrtlist(__p(self.camXs_T_camRs), __p(lrtlist_camRs)))
        # these are is B x S x N x 19

        self.summ_writer.summ_lrtlist('obj/lrtlist_camX0', self.rgb_camXs[:,0], lrtlist_camXs[:,0],
                                      scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        self.summ_writer.summ_lrtlist('obj/lrtlist_camR0', self.rgb_camRs[:,0], lrtlist_camRs[:,0],
                                      scorelist_s[:,0], tidlist_s[:,0], self.pix_T_cams[:,0])
        # mask_memX0 = utils.vox.assemble_padded_obj_masklist(
        #     lrtlist_camXs[:,0], scorelist_s[:,0], self.Z2, self.Y2, self.X2, coeff=1.0)
        # mask_memX0 = torch.sum(mask_memX0, dim=1).clamp(0, 1) 
        # self.summ_writer.summ_oned('obj/mask_memX0', mask_memX0, bev=True)

        mask_memXs = __u(utils.vox.assemble_padded_obj_masklist(
            __p(lrtlist_camXs), __p(scorelist_s), self.Z2, self.Y2, self.X2, coeff=1.0))
        mask_memXs = torch.sum(mask_memXs, dim=2).clamp(0, 1)
        self.summ_writer.summ_oneds('obj/mask_memXs', torch.unbind(mask_memXs, dim=1), bev=True)

        for b in list(range(self.B)):
            for s in list(range(self.S)):
                mask = mask_memXs[b,s]
                if torch.sum(mask) < 2.0:
                    # return early
                    return total_loss, None, True
                
        # next: i want to treat features differently if they are in obj masks vs not
        # in particular, i want a different kind of retrieval metric
        
        if hyp.do_feat3d:
            # occXs is B x S x 1 x H x W x D
            # unpXs is B x S x 3 x H x W x D
            feat_memXs_input = torch.cat([self.occXs, self.occXs*self.unpXs], dim=2)
            feat_memXs_input_ = __p(feat_memXs_input)

            feat_memXs_, _, _ = self.feat3dnet(
                feat_memXs_input_,
                self.summ_writer,
                comp_mask=None,
            )
            feat_memXs = __u(feat_memXs_)
                                    
            self.summ_writer.summ_feats('3d_feats/feat_memXs_input', torch.unbind(feat_memXs_input, dim=1), pca=True)
            self.summ_writer.summ_feats('3d_feats/feat_memXs_output', torch.unbind(feat_memXs, dim=1), pca=True)

            mv_precision = utils.eval.measure_semantic_retrieval_precision(feat_memXs[0], mask_memXs[0])
            self.summ_writer.summ_scalar('semantic_retrieval/multiview_precision', mv_precision)
            ms_precision = utils.eval.measure_semantic_retrieval_precision(feat_memXs[:,0], mask_memXs[:,0])
            self.summ_writer.summ_scalar('semantic_retrieval/multiscene_precision', ms_precision)
            
        return total_loss, None, False
            
    def forward(self, feed):
        data_ok = self.prepare_common_tensors(feed)

        if not data_ok:
            # return early
            total_loss = torch.tensor(0.0).cuda()
            return total_loss, None, True
        else:
            if self.set_name=='train':
                return self.run_train(feed)
            elif self.set_name=='test':
                return self.run_test(feed)
            else:
                print('weird set_name:', set_name)
                assert(False)
