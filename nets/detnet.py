import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.ops as ops

import utils.basic
import utils.geom
import utils.misc
import hyperparams as hyp
import archs.encoder3d

# import utils.misc

# class HYP_debug(object):
#     def __init__(self):
#         self.det_anchor_size=1.0

# hyp = HYP_debug()

def smooth_l1_loss(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs = deltas - targets

    smooth_l1_signs = (torch.abs(diffs) < 1.0 / sigma2).float() 

    smooth_l1_option1 = diffs**2 * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = smooth_l1_option1 * smooth_l1_signs + smooth_l1_option2 * (1 - smooth_l1_signs)
    smooth_l1 = smooth_l1_add

    return smooth_l1

def binarize(input, threshold):
    return torch.where(input < threshold, torch.zeros_like(input), torch.ones_like(input))

def meshgrid3d_xyz(B, Z, Y, X):
    grid_z, grid_y, grid_x = utils.basic.meshgrid3d(B, Z, Y, X, stack=False)
    # each one is shaped B x Z x Y x X
    grid_z = grid_z.permute(0, 3, 2, 1)
    grid_x = grid_x.permute(0, 3, 2, 1)
    grid_y = grid_y.permute(0, 3, 2, 1)
    # make each one axis order XYZ
    grid = torch.stack([grid_x, grid_y, grid_z], dim=-1)
    
    return grid

def anchor_deltas_to_bboxes(anchor_deltas, indices):
    # anchors deltas is num_objects x 6, first 3 for translation and last 3 for scale
    # grid_center is num_objects x 3

    grid_center = indices.float()
    object_center = grid_center + anchor_deltas[:, :3] * hyp.det_anchor_size
    object_min = object_center - 0.5 * torch.exp(anchor_deltas[:, 3:]) * hyp.det_anchor_size
    object_max = object_center + 0.5 * torch.exp(anchor_deltas[:, 3:]) * hyp.det_anchor_size
    return torch.stack([object_min, object_max], 2), torch.cat([object_center, object_max - object_min], 1) #these are N x 3 x 2 and N x 6, respectively

def overlap_graph(boxes1, boxes2): #tested
    # boxes1: batch x 3 x 2 (z1,z2,y1,y2,x1,x2)
    b1_bs = boxes1.shape[0] #batch_size
    b2_bs = boxes2.shape[0]

    if b1_bs == 0 or b2_bs == 0:
        # torch's repeat will fail, so let's return early
        return torch.zeros(b1_bs, b2_bs)

    boxes1 = boxes1.view(-1, 6)
    boxes2 = boxes2.view(-1, 6)

    b1 = boxes1.unsqueeze(1).repeat(1, b2_bs, 1).view(-1, 6) #this is (b1xb2) x 6
    b2 = boxes2.unsqueeze(0).repeat(b1_bs, 1, 1).view(-1, 6)

    b1_z1, b1_z2, b1_y1, b1_y2, b1_x1, b1_x2 = torch.chunk(b1, 6, dim=1)
    b2_z1, b2_z2, b2_y1, b2_y2, b2_x1, b2_x2 = torch.chunk(b2, 6, dim=1)

    z1 = torch.max(b1_z1, b2_z1)
    z2 = torch.min(b1_z2, b2_z2)
    y1 = torch.max(b1_y1, b2_y1)
    y2 = torch.min(b1_y2, b2_y2)
    x1 = torch.max(b1_x1, b2_x1)
    x2 = torch.min(b1_x2, b2_x2)

    intersection = torch.max(z2 - z1, torch.zeros_like(z1)) * torch.max(y2 - y1, torch.zeros_like(y1)) * torch.max(x2 - x1, torch.zeros_like(x1))

    b1_area = (b1_z2 - b1_z1) * (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_z2 - b2_z1) * (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection

    iou = intersection / union
    overlaps = iou.view(b1_bs, b2_bs)

    return overlaps

def box_refinement_graph(positive_rois, roi_gt_boxes):
    # roi_gt_boxes are N x 3 x 2
    gt_center = torch.mean(roi_gt_boxes, dim=2)
    pd_center = torch.mean(positive_rois, dim=2) #these are N x 3 (zyx order)

    delta_zyx = gt_center-pd_center

    len_gt = roi_gt_boxes[:,:,1] - roi_gt_boxes[:,:,0]
    len_pd = positive_rois[:,:,1] - positive_rois[:,:,0]

    delta_len = len_gt - len_pd

    return torch.cat([delta_zyx, delta_len], dim=1) # N x 6

def rpn_proposal_graph(pred_objectness, pred_anchor_deltas, valid_mask, corners_min_max_g, iou_thresh=0.5): #tested
    ######################## ROI generation ####################
    # object_bbox: batch_size x N x 3 x 2
    # pred_objectness: B x X x Y x Z
    # pred_anchor_deltas: B x X x Y x Z x 6
    # valid_mask: B x N
    # corners_min_max_g: B x N x 3 x 2, in xyz order

    P_THRES = 0.9
    high_prob_indices = torch.stack(torch.where(pred_objectness > P_THRES), dim=1) # this is ? x 4, last dim in bxyz order
    B = pred_objectness.shape[0]

    # build prediction target
    bs_selected_boxes_co = []
    bs_selected_scores = []
    bs_overlaps = []
    if len(high_prob_indices > 0):
        for i in list(range(B)):
            selected_boxes, selected_boxes_scores, overlaps, selected_boxes_co = detection_target_graph(i, high_prob_indices, \
            corners_min_max_g, valid_mask, pred_objectness, pred_anchor_deltas, iou_thresh=iou_thresh)

            bs_selected_boxes_co.append(selected_boxes_co)
            bs_selected_scores.append(selected_boxes_scores)
            bs_overlaps.append(overlaps)
        return bs_selected_boxes_co, bs_selected_scores, bs_overlaps
    else:
        return None, None, None

# def gather_nd(tensor, index): # pls dont use this, this doesn't work...
#     index_shape = list(index.shape)
#     tensor_shape = list(tensor.shape)

#     axis_dim = index_shape[-1]
#     assert axis_dim <= tensor.ndim

#     final_shape = index_shape[:-1] + tensor_shape[axis_dim:]

#     print(axis_dim)

#     index = index.view(-1, axis_dim) # flatten it
#     res = []

#     for i in index:
#         res.append(tuple(tensor[i]))

#     res = torch.stack(res, dim=0) 

#     return res.view(final_shape)


def detection_target_graph(i, high_prob_indices, corners_min_max_g, valid_mask, pred_objectness, pred_anchor_deltas,
                           iou_thresh=0.5): #tested

    batch_i_idxs = torch.stack(torch.where(high_prob_indices[:,0] == i), dim=1) # this is (?, 1)
    batch_i_indices = high_prob_indices[batch_i_idxs.squeeze(dim=1)] # this is ? x 4

    batch_i_scores = pred_objectness[batch_i_indices[:, 0], batch_i_indices[:, 1], batch_i_indices[:, 2], batch_i_indices[:, 3]] # this is (?, )
    batch_i_anchor_deltas = pred_anchor_deltas[batch_i_indices[:, 0], batch_i_indices[:, 1], batch_i_indices[:, 2], batch_i_indices[:, 3]] # this is (?, 6)

    # don't know why all out of a sudden order becomes zyx, but we follow this zyx order for the following code ...

    # co refers to center + offset parameterization
    batch_i_bboxes, batch_i_bboxes_co = anchor_deltas_to_bboxes(
        batch_i_anchor_deltas, batch_i_indices[:,1:]) 
    # N x 3 x 2 and N x 6

    # print(batch_i_bboxes[:, 1:, :].permute(0, 2, 1).shape)

    selected_bboxes_idx_xy = ops.nms(
        batch_i_bboxes[:, 1:, :].permute(0, 2, 1).contiguous().view(-1, 4).cpu(), # view() fails, so we introduce this contiguous()
        batch_i_scores.cpu(),
        iou_thresh).cuda()
    selected_bboxes_idx_zx = ops.nms(
        batch_i_bboxes[:, [0,2], :].permute(0, 2, 1).contiguous().view(-1, 4).cpu(),
        batch_i_scores.cpu(),
        iou_thresh).cuda()
    selected_bboxes_idx = torch.unique(torch.cat([selected_bboxes_idx_xy, selected_bboxes_idx_zx], dim=0)) # this is (selected_bbox, )

    selected_3d_bboxes = batch_i_bboxes[selected_bboxes_idx] # this is (selected_bbox, 3, 2)

    selected_3d_bboxes_co = batch_i_bboxes_co[selected_bboxes_idx] # this is (selected_bbox, 6)
    selected_3d_bboxes_scores = batch_i_scores[selected_bboxes_idx]
    valid_inds = torch.stack(torch.where(valid_mask[i, :]), dim=1).squeeze(dim=1) # this is (valid_ids, )
    corners_min_max_g_i = corners_min_max_g[i, valid_inds] # (valid_ids, 3, 2)

    # calculate overlap in 3d
    overlaps = overlap_graph(selected_3d_bboxes, corners_min_max_g_i) # this is (selected_bbox, valid_ids)

    return selected_3d_bboxes, selected_3d_bboxes_scores, overlaps, selected_3d_bboxes_co

    # # calculate overlap in 3d
    # overlaps = overlap_graph(selected_3d_bboxes, corners_min_max_g_i) # this is (selected_bbox, valid_ids)
    # roi_iou_max = torch.max(overlaps, dim=1)[0] # (selected_bbox, )
    # positive_roi_bool = (roi_iou_max >= 0.5)
    # positive_indices = torch.where(positive_roi_bool)[0]
    # negative_roi_bool = (roi_iou_max < 0.5)
    # negative_indices = torch.where(negative_roi_bool)[0]

    # positive_count = 3 # [what is this??]

    # positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    # positive_count = tf.shape(positive_indices)[0]
    # # Negative ROIs. Add enough to maintain positive:negative ratio.
    # ratio = 1.0 / 0.3
    # negative_count = tf.cast(ratio * tf.cast(positive_count, tf.float32), tf.int32) - positive_count
    # negative_indices = tf.random_shuffle(negative_indices)[:negative_count]
    # # Gather selected ROIs
    # positive_rois = tf.gather(selected_3d_bboxes, positive_indices)
    # positive_rois_score = tf.gather(selected_3d_bboxes_scores, positive_indices)
    # negative_rois = tf.gather(selected_3d_bboxes, negative_indices)

    # # Assign positive ROIs to GT boxes.
    # positive_overlaps = tf.gather(overlaps, positive_indices)
    # roi_gt_box_assignment = tf.cond(
    #     tf.greater(tf.shape(positive_overlaps)[1], 0),
    #     true_fn = lambda: tf.argmax(positive_overlaps, axis=1),
    #     false_fn = lambda: tf.cast(tf.constant([]),tf.int64)
    # )
    # # corners_min_max_g_i is N x 3 x 2 (zyx, zyx)
    # roi_gt_boxes = tf.gather(corners_min_max_g_i, roi_gt_box_assignment)
    # # nrois x 128 x 128 x 128

    # deltas = box_refinement_graph(positive_rois, roi_gt_boxes)
    # rois = tf.concat([positive_rois, negative_rois], axis=0)
    # class_label = tf.ones((tf.shape(positive_rois)[0]))
    # N_neg = tf.shape(negative_rois)[0]
    # N_pos = tf.maximum(10 - tf.shape(rois)[0], 0)
    # class_label = tf.pad(class_label, [(0, N_neg+N_pos)])
    # rois = tf.pad(rois, [(0, N_pos), (0, 0), (0, 0)])
    # deltas = tf.pad(deltas, [(0, N_neg + N_pos), (0, 0)])
    # return rois, deltas, class_label, selected_3d_bboxes, selected_3d_bboxes_scores, overlaps, selected_3d_bboxes_co

class DetNet(nn.Module):
    def __init__(self):
        print('DetNet...')

        super(DetNet, self).__init__()
        self.pred_dim = 7
        
        in_dim = 4
        # self.net = archs.encoder3d.Net3d(in_channel=in_dim, pred_dim=self.pred_dim).cuda()
        self.net = torch.nn.Conv3d(in_channels=hyp.feat3d_dim, out_channels=self.pred_dim, kernel_size=3, stride=1, padding=1).cuda()
        print(self.net)

    def forward(self,
                lrtlist_g,
                scores_g,
                feat_zyx,
                summ_writer
    ):
        
        total_loss = torch.tensor(0.0).cuda()

        B, C, Z, Y, X = feat_zyx.shape
        _, N, _ = lrtlist_g.shape
        # Z, Y, X = int(Z/2), int(Y/2), int(X/2)

        total_loss = 0.0
        pred_dim = self.pred_dim # total 7, 6 deltas, 1 objectness

        feat = feat_zyx.permute(0, 1, 4, 3, 2) # get feat in xyz order, now B x C x X x Y x Z

        corners = utils.geom.get_xyzlist_from_lrtlist(lrtlist_g) # corners is B x N x 8 x 3, last dim in xyz order
        corners_max = torch.max(corners, dim=2)[0] # B x N x 3
        corners_min = torch.min(corners, dim=2)[0]
        corners_min_max_g = torch.stack([corners_min, corners_max], dim=3) # this is B x N x 3 x 2

        # trim down, to save some time
        N = min(N, hyp.K)
        corners_min_max_g = corners_min_max_g[:,:N]
        scores_g = scores_g[:, :N] # B x N

        # boxes_g is [-0.5~63.5, -0.5~15.5, -0.5~63.5]
        centers_g = utils.geom.get_clist_from_lrtlist(lrtlist_g)
        # centers_g is B x N x 3
        grid = meshgrid3d_xyz(B, Z, Y, X)[0] # just one grid please, this is X x Y x Z x 3

        delta_positions_raw = centers_g.view(B, N, 1, 1, 1, 3) - grid.view(1, 1, X, Y, Z, 3)
        # tf.summary.histogram('delta_positions_raw', delta_positions_raw)
        delta_positions = delta_positions_raw / hyp.det_anchor_size
        # tf.summary.histogram('delta_positions', delta_positions)

        lengths_g = utils.geom.get_lenlist_from_lrtlist(lrtlist_g) # B x N x 3
        # tf.summary.histogram('lengths_g', lengths_g)
        delta_lengths = torch.log(lengths_g / hyp.det_anchor_size)
        delta_lengths = torch.max(delta_lengths, -1e6 * torch.ones_like(delta_lengths)) # to avoid -infs turning into nans
        # tf.summary.histogram('delta_lengths', delta_lengths)

        lengths_g = lengths_g.view(B, N, 1, 1, 1, 3).repeat(1, 1, X, Y, Z, 1) # B x N x X x Y x Z x 3
        delta_lengths = delta_lengths.view(B, N, 1, 1, 1, 3).repeat(1, 1, X, Y, Z, 1) # B x N x X x Y x Z x 3
        valid_mask = scores_g.view(B, N, 1, 1, 1, 1).repeat(1, 1, X, Y, Z, 1) # B x N x X x Y x Z x 1

        delta_gt = torch.cat([delta_positions, delta_lengths], -1) # B x N x X x Y x Z x 6

        object_dist = torch.max(torch.abs(delta_positions_raw)/(lengths_g * 0.5 + 1e-5), dim=5)[0] # B x N x X x Y x Z
        object_dist_mask = (torch.ones_like(object_dist) - binarize(object_dist, 0.5)).unsqueeze(dim=5) # B x N x X x Y x Z x 1
        object_dist_mask = object_dist_mask * valid_mask # B x N x X x Y x Z x 1
        object_neg_dist_mask = torch.ones_like(object_dist) - binarize(object_dist, 0.8)
        object_neg_dist_mask = object_neg_dist_mask * valid_mask.squeeze(dim=5) # B x N x X x Y x Z

        anchor_deltas_gt = None
        for obj_id in list(range(N)):
            if anchor_deltas_gt is None:
                anchor_deltas_gt = delta_gt[:, obj_id, :, :, :, :] * object_dist_mask[:, obj_id, :, :, :, :]
                current_mask = object_dist_mask[:, obj_id, :, :, :, :]

            else:
                # don't overwrite anchor positions that are already taken
                overlap = current_mask * object_dist_mask[:, obj_id, :, :, :, :]
                anchor_deltas_gt += (torch.ones_like(overlap)- overlap) * delta_gt[:, obj_id, :, :, :, :] * object_dist_mask[:, obj_id, :, :, :, :]
                current_mask = current_mask + object_dist_mask[:, obj_id, :, :, :, :]
                current_mask = binarize(current_mask,  0.5)

        # tf.summary.histogram('anchor_deltas_gt', anchor_deltas_gt)
        # ok nice, these do not have any extreme values

        

        pos_equal_one = binarize(torch.sum(object_dist_mask, dim=1), 0.5).squeeze(dim=4) # B x X x Y x Z
        neg_equal_one = binarize(torch.sum(object_neg_dist_mask, dim=1), 0.5) 
        neg_equal_one = torch.ones_like(neg_equal_one) - neg_equal_one # B x X x Y x Z
        pos_equal_one_sum = torch.sum(pos_equal_one, [1,2,3]) # B
        neg_equal_one_sum = torch.sum(neg_equal_one, [1,2,3])

        summ_writer.summ_occ('det/pos_equal_one', pos_equal_one.unsqueeze(1))

        # set min to one in case no object, to avoid nan
        pos_equal_one_sum_safe = torch.max(pos_equal_one_sum, torch.ones_like(pos_equal_one_sum)) # B
        neg_equal_one_sum_safe = torch.max(neg_equal_one_sum, torch.ones_like(neg_equal_one_sum)) # B

        pred = self.net(feat) # this is B x 7 x X x Y x Z
        summ_writer.summ_feat('det/feat', feat, pca=False)
        summ_writer.summ_feat('det/pred', pred, pca=True)
        
        # print('feat', feat.shape)
        # print('pred', pred.shape)
        pred = pred.permute(0, 2, 3, 4, 1) # B x X x Y x Z x 7
        pred_anchor_deltas = pred[..., 1:] # B x X x Y x Z x 6
        pred_objectness_logits = pred[..., 0] # B x X x Y x Z
        pred_objectness = torch.nn.functional.sigmoid(pred_objectness_logits) # B x X x Y x Z

        # pred_anchor_deltas = pred_anchor_deltas.cpu()
        # pred_objectness = pred_objectness.cpu()

        alpha = 1.5
        beta = 1.0
        small_addon_for_BCE = 1e-6

        overall_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            input=pred_objectness_logits,
            target=pos_equal_one,
            reduction='none',
        )
        cls_pos_loss = utils.basic.reduce_masked_mean(overall_loss, pos_equal_one)
        cls_neg_loss = utils.basic.reduce_masked_mean(overall_loss, neg_equal_one)
        loss_prob = torch.sum(alpha * cls_pos_loss + beta * cls_neg_loss)

        pos_mask = pos_equal_one.unsqueeze(dim=4) # B x X x Y x Z x 1
        loss_l1 = smooth_l1_loss(pos_mask * pred_anchor_deltas, pos_mask * anchor_deltas_gt) # B x X x Y x Z x 1
        loss_reg = torch.sum(loss_l1/pos_equal_one_sum_safe.view(-1, 1, 1, 1, 1))/float(B)

        total_loss = utils.misc.add_loss('det/detect_prob', total_loss, loss_prob, hyp.det_prob_coeff, summ_writer)
        total_loss = utils.misc.add_loss('det/detect_reg', total_loss, loss_reg, hyp.det_reg_coeff, summ_writer)

        # finally, turn the preds into hard boxes, with nms
        (
            bs_selected_boxes_co,
            bs_selected_scores,
            bs_overlaps,
        ) = rpn_proposal_graph(pred_objectness, pred_anchor_deltas, scores_g, corners_min_max_g,
                               iou_thresh=0.2)
        # these are lists of length B, each one leading with dim "?", since there is a variable number of objs per frame

        N = hyp.K*2
        tidlist = torch.linspace(1.0, N, N).long().to('cuda')
        tidlist = tidlist.unsqueeze(0).repeat(B, 1)
        padded_boxes_e = torch.zeros(B, N, 9).float().cuda()
        padded_scores_e = torch.zeros(B, N).float().cuda()
        if bs_selected_boxes_co is not None: 

            for b in list(range(B)):
                
                # make the boxes 1 x N x 9 (instead of B x ? x 6)
                padded_boxes0_e = bs_selected_boxes_co[b].unsqueeze(0)
                padded_scores0_e = bs_selected_scores[b].unsqueeze(0)

                padded_boxes0_e = torch.cat([padded_boxes0_e, torch.zeros([1, N, 6], device=torch.device('cuda'))], dim=1) # 1 x ? x 6
                padded_scores0_e = torch.cat([padded_scores0_e, torch.zeros([1, N], device=torch.device('cuda'))], dim=1) # pad out

                padded_boxes0_e = padded_boxes0_e[:,:N] # clip to N
                padded_scores0_e = padded_scores0_e[:,:N] # clip to N

                padded_boxes0_e = torch.cat([padded_boxes0_e, torch.zeros([1, N, 3], device=torch.device('cuda'))], dim=2)

                padded_boxes_e[b] = padded_boxes0_e[0]
                padded_scores_e[b] = padded_scores0_e[0]
        return total_loss, padded_boxes_e, padded_scores_e, tidlist, pred_objectness, bs_selected_scores, bs_overlaps

if __name__ == "__main__":
    A = torch.randn(5, 10)
    B = torch.randn(5, 10)
    # print(smooth_l1_loss(A, A+1))
    # meshgrid3d_xyz(2, 64, 64, 64)

    boxes1 = torch.randn(2, 3, 1)
    boxes1 = boxes1.repeat(1, 1, 2) #2 x 3 x 2
    boxes1[:, :, 1] += 1.0
    boxes2 = boxes1 - 0.5

    # print(overlap_graph(boxes1, boxes2))
    # print(box_refinement_graph(boxes1, boxes2))

    # boxes3d = torch.zeros(2, 2, 9).cuda()

    pred_objectness = torch.zeros(2, 10, 10, 10)
    pred_objectness[0,1,1,1] = 1.0
    pred_anchor_deltas = torch.zeros(2, 10, 10, 10, 6)
    valid_mask = torch.ones(2, 1)
    corners_min_max_g = torch.tensor(np.array([[0.0, 1.5], [0.0, 1.5], [0.5, 1.5]])).view(1, 1, 3, 2).repeat(2, 1, 1, 1).float()

    bs_selected_boxes_co, bs_selected_scores, bs_overlaps = rpn_proposal_graph(pred_objectness, pred_anchor_deltas, valid_mask, corners_min_max_g)
    print(bs_overlaps)
    




