import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("..")

import archs.encoder2d as encoder2d
import hyperparams as hyp
import utils.basic
import utils.misc
import utils.improc

class Emb2dNet(nn.Module):
    def __init__(self):
        super(Emb2dNet, self).__init__()

        print('Emb2dNet...')
        self.batch_k = 2
        self.num_samples = hyp.emb2d_num_samples
        assert(self.num_samples > 0)
        self.sampler = utils.misc.DistanceWeightedSampling(batch_k=self.batch_k, normalize=False)
        self.criterion = utils.misc.MarginLoss() #margin=args.margin,nu=args.nu)
        self.beta = 1.2

        self.dict_len = 20000
        self.neg_pool = utils.misc.SimplePool(self.dict_len, version='pt')
        self.ce = torch.nn.CrossEntropyLoss()
        
    def sample_embs(self, emb0, emb1, valid, B, Y, X, mod='', do_vis=False, summ_writer=None):
        if hyp.emb2d_mindist == 0.0:
            # pure random
            perm = torch.randperm(B*Y*X)
            emb0 = emb0.reshape(B*Y*X, -1)
            emb1 = emb1.reshape(B*Y*X, -1)
            valid = valid.reshape(B*Y*X, -1)
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
                    valid[b], (Y, X), self.num_samples, mode='2d', tol=hyp.emb2d_mindist)
                emb0_s_ = emb0[b, sample_indices]
                emb1_s_ = emb1[b, sample_indices]
                # these are N x D
                emb0_all.append(emb0_s_)
                emb1_all.append(emb1_s_)
                valid_all.append(sample_valids)

            if do_vis and (summ_writer is not None):
                sample_mask = utils.improc.xy2mask_single(sample_locs, Y, X)
                summ_writer.summ_oned('emb2d/samples_%s/sample_mask' % mod, torch.unsqueeze(sample_mask, dim=0))
                summ_writer.summ_oned('emb2d/samples_%s/valid' % mod, torch.reshape(valid, [B, 1, Y, X]))
            emb0_all = torch.cat(emb0_all, axis=0)
            emb1_all = torch.cat(emb1_all, axis=0)
            valid_all = torch.cat(valid_all, axis=0)
            return emb0_all, emb1_all, valid_all

    def compute_margin_loss(self, B, C, Y, X, emb0_vec, emb1_vec, valid_vec, mod='', do_vis=False, summ_writer=None):
        emb0_vec, emb1_vec, valid_vec = self.sample_embs(
            emb0_vec,
            emb1_vec,
            valid_vec,
            B, Y, X,
            mod=mod,
            do_vis=do_vis,
            summ_writer=summ_writer)
        
        emb_vec = torch.stack((emb0_vec, emb1_vec), dim=1).view(B*self.num_samples*self.batch_k,C)
        # this tensor goes e,g,e,g,... on dim 0
        # note this means 2 samples per class; batch_k=2
        y = torch.stack([torch.arange(0,self.num_samples*B), torch.arange(0,self.num_samples*B)], dim=1).view(self.num_samples*B*self.batch_k)
        # this tensor goes 0,0,1,1,2,2,...

        a_indices, anchors, positives, negatives, _ = self.sampler(emb_vec)
        margin_loss, _ = self.criterion(anchors, positives, negatives, self.beta, y[a_indices])
        return margin_loss

    def compute_ce_loss(self, B, C, Y, X, emb_e_vec_all, emb_g_vec_all, valid_vec_all, mod='', do_vis=False, summ_writer=None):
        emb_e_vec, emb_g_vec, valid_vec = self.sample_embs(emb_e_vec_all,
                                                           emb_g_vec_all,
                                                           valid_vec_all,
                                                           B, Y, X,
                                                           mod=mod,
                                                           do_vis=do_vis,
                                                           summ_writer=summ_writer)
        _, emb_n_vec, _ = self.sample_embs(emb_e_vec_all,
                                           emb_g_vec_all,
                                           valid_vec_all,
                                           B, Y, X,
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
    
    def forward(self, emb_e, emb_g, valid, summ_writer=None, suffix=''):
        total_loss = torch.tensor(0.0).cuda()

        if torch.isnan(emb_e).any() or torch.isnan(emb_g).any():
            assert(False)

        B, C, H, W = list(emb_e.shape)
        # put channels on the end
        emb_e_vec = emb_e.permute(0,2,3,1).reshape(B, H*W, C)
        emb_g_vec = emb_g.permute(0,2,3,1).reshape(B, H*W, C)
        valid_vec = valid.permute(0,2,3,1).reshape(B, H*W, 1)
        
        assert(self.num_samples < (B*H*W))
        # we will take num_samples from each one

        margin_loss = self.compute_margin_loss(B, C, H, W, emb_e_vec, emb_g_vec, valid_vec, 'all', True, summ_writer)
        total_loss = utils.misc.add_loss('emb2d/emb2d_ml_loss%s' % suffix, total_loss, margin_loss, hyp.emb2d_ml_coeff, summ_writer)

        ce_loss = self.compute_ce_loss(B, C, H, W, emb_e_vec, emb_g_vec.detach(), valid_vec, 'g', False, summ_writer)
        total_loss = utils.misc.add_loss('emb2d/emb_ce_loss', total_loss, ce_loss, hyp.emb2d_ce_coeff, summ_writer)
        
        l2_loss_im = utils.basic.sql2_on_axis(emb_e-emb_g.detach(), 1, keepdim=True)
        emb_l2_loss = utils.basic.reduce_masked_mean(l2_loss_im, valid)
        total_loss = utils.misc.add_loss('emb2d/emb2d_l2_loss%s' % suffix, total_loss, emb_l2_loss, hyp.emb2d_l2_coeff, summ_writer)

        if summ_writer is not None:
            summ_writer.summ_oned('emb2d/emb2d_l2_loss%s' % suffix, l2_loss_im)
            summ_writer.summ_feats('emb2d/embs_2d%s' % suffix, [emb_e, emb_g], pca=True)

        return total_loss, emb_g

    
