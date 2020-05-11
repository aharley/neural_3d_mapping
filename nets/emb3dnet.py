import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import hyperparams as hyp
import utils.improc
import utils.misc
import utils.vox
import utils.basic

class Emb3dNet(nn.Module):
    def __init__(self):
        super(Emb3dNet, self).__init__()

        print('Emb3dNet...')
        self.batch_k = 2
        self.num_samples = hyp.emb3d_num_samples
        assert(self.num_samples > 0)
        self.sampler = utils.misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False)
        self.criterion = utils.misc.MarginLoss() #margin=args.margin,nu=args.nu)
        self.beta = 1.2

        self.dict_len = 20000
        self.neg_pool = utils.misc.SimplePool(self.dict_len, version='pt')
        self.ce = torch.nn.CrossEntropyLoss()

    def sample_embs(self, emb0, emb1, valid, B, Z, Y, X, mod='', do_vis=False, summ_writer=None):
        if hyp.emb3d_mindist == 0.0:
            # pure random
            perm = torch.randperm(B*Z*Y*X)
            emb0 = emb0.reshape(B*Z*Y*X, -1)
            emb1 = emb1.reshape(B*Z*Y*X, -1)
            valid = valid.reshape(B*Z*Y*X, -1)
            emb0 = emb0[perm[:self.num_samples*B]]
            emb1 = emb1[perm[:self.num_samples*B]]
            valid = valid[perm[:self.num_samples*B]]
            return emb0, emb1, valid
        else:
            emb0_all = []
            emb1_all = []
            valid_all = []
            for b in list(range(B)):
                sample_indices, sample_locs, sample_valids = utils.misc.get_safe_samples(
                    valid[b], (Z, Y, X), self.num_samples, mode='3d', tol=hyp.emb3d_mindist)
                emb0_s_ = emb0[b, sample_indices]
                emb1_s_ = emb1[b, sample_indices]
                # these are N x D
                emb0_all.append(emb0_s_)
                emb1_all.append(emb1_s_)
                valid_all.append(sample_valids)

            if do_vis and (summ_writer is not None):
                sample_occ = utils.vox.voxelize_xyz(torch.unsqueeze(sample_locs, dim=0), Z, Y, X, already_mem=True)
                summ_writer.summ_occ('emb3d/samples_%s/sample_occ' % mod, sample_occ, reduce_axes=[2,3])
                summ_writer.summ_occ('emb3d/samples_%s/valid' % mod, torch.reshape(valid, [B, 1, Z, Y, X]), reduce_axes=[2,3])

            emb0_all = torch.cat(emb0_all, axis=0)
            emb1_all = torch.cat(emb1_all, axis=0)
            valid_all = torch.cat(valid_all, axis=0)
            return emb0_all, emb1_all, valid_all
        
    def compute_ce_loss(self, B, C, Z, Y, X, emb_e_vec_all, emb_g_vec_all, valid_vec_all, mod='', do_vis=False, summ_writer=None):
        emb_e_vec, emb_g_vec, valid_vec = self.sample_embs(emb_e_vec_all,
                                                           emb_g_vec_all,
                                                           valid_vec_all,
                                                           B, Z, Y, X,
                                                           mod=mod,
                                                           do_vis=do_vis,
                                                           summ_writer=summ_writer)
        _, emb_n_vec, _ = self.sample_embs(emb_e_vec_all,
                                           emb_g_vec_all,
                                           valid_vec_all,
                                           B, Z, Y, X,
                                           mod=mod,
                                           do_vis=do_vis,
                                           summ_writer=summ_writer)
        emb_e_vec = emb_e_vec.view(B*self.num_samples, C)
        emb_g_vec = emb_g_vec.view(B*self.num_samples, C)
        emb_n_vec = emb_n_vec.view(B*self.num_samples, C)
        
        self.neg_pool.update(emb_n_vec.cpu())
        # print('neg_pool len:', len(self.neg_pool))
        emb_n = self.neg_pool.fetch().cuda()

        # print('emb_n', emb_n.shape)
        N2, C2 = list(emb_n.shape)
        assert (C2 == C)
        
        # l_negs = torch.mm(q.view(N, C), negs.view(C, N2)) # this is N x N2

        emb_q = emb_e_vec.clone()
        emb_k = emb_g_vec.clone()
        
        # print('emb_q', emb_q.shape)
        # print('emb_k', emb_k.shape)
        N = emb_q.shape[0]
        l_pos = torch.bmm(emb_q.view(N,1,-1), emb_k.view(N,-1,1))

        # print('l_pos', l_pos.shape)
        l_neg = torch.mm(emb_q, emb_n.T)
        # print('l_neg', l_neg.shape)
        
        l_pos = l_pos.view(N, 1)
        # print('l_pos', l_pos.shape)
        logits = torch.cat([l_pos, l_neg], dim=1)

        labels = torch.zeros(N, dtype=torch.long).cuda()

        temp = 0.07
        emb_loss = self.ce(logits/temp, labels)
        # print('emb_loss', emb_loss.detach().cpu().numpy())
        return emb_loss
            
    def forward(self, emb_e, emb_g, vis_e, vis_g, summ_writer=None):
        total_loss = torch.tensor(0.0).cuda()

        if torch.isnan(emb_e).any() or torch.isnan(emb_g).any():
            assert(False)

        B, C, D, H, W = list(emb_e.shape)
        # put channels on the end
        emb_e_vec = emb_e.permute(0,2,3,4,1).reshape(B, D*H*W, C)
        emb_g_vec = emb_g.permute(0,2,3,4,1).reshape(B, D*H*W, C)
        vis_e_vec = vis_e.permute(0,2,3,4,1).reshape(B, D*H*W, 1)
        vis_g_vec = vis_g.permute(0,2,3,4,1).reshape(B, D*H*W, 1)

        # ensure they are both nonzero, else we probably masked or warped something
        valid_vec_e = 1.0 - (emb_e_vec==0).all(dim=2, keepdim=True).float()
        valid_vec_g = 1.0 - (emb_g_vec==0).all(dim=2, keepdim=True).float()
        valid_vec = valid_vec_e * valid_vec_g
        vis_e_vec *= valid_vec
        vis_g_vec *= valid_vec
        # valid_g = 1.0 - (emb_g==0).all(dim=1, keepdim=True).float()
        
        assert(self.num_samples < (B*D*H*W))
        # we will take num_samples from each one

        ce_loss = self.compute_ce_loss(B, C, D, H, W, emb_e_vec, emb_g_vec.detach(), vis_g_vec, 'g', False, summ_writer)
        total_loss = utils.misc.add_loss('emb3d/emb_ce_loss', total_loss, ce_loss, hyp.emb3d_ce_coeff, summ_writer)

        # where g is valid, we use it as reference and pull up e
        l2_loss = utils.basic.reduce_masked_mean(utils.basic.sql2_on_axis(emb_e-emb_g.detach(), 1, keepdim=True), vis_g)
        total_loss = utils.misc.add_loss('emb3d/emb3d_l2_loss', total_loss, l2_loss, hyp.emb3d_l2_coeff, summ_writer)

        l2_loss_im = torch.mean(utils.basic.sql2_on_axis(emb_e-emb_g, 1, keepdim=True), dim=3)
        if summ_writer is not None:
            summ_writer.summ_oned('emb3d/emb3d_l2_loss', l2_loss_im)
            summ_writer.summ_feats('emb3d/embs_3d', [emb_e, emb_g], pca=True)
        return total_loss

