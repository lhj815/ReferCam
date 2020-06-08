import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops.roi_align as roi_align

from matplotlib import pyplot as plt

from PIL import Image
import numpy as np
import random
from collections import OrderedDict
from .darknet import *

from utils.parsing_metrics import *
from utils.utils import *
from train_yolo import get_args


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def bbox_transform_inv(boxes, deltas, batch_size,grid_size):
    boxes=boxes/grid_size

    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = F.sigmoid(dx) * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = F.sigmoid(dy) * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes*grid_size

def clip_boxes(boxes, im_shape, batch_size):

    for i in range(batch_size):
        boxes[i,:,0::4].clamp_(0, im_shape-1)
        boxes[i,:,1::4].clamp_(0, im_shape-1)
        boxes[i,:,2::4].clamp_(0, im_shape-1)
        boxes[i,:,3::4].clamp_(0, im_shape-1)

    return boxes

def bbox_overlaps_batch(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)


    if anchors.dim() == 2:

        N = anchors.size(0)
        K = gt_boxes.size(1)

        anchors = anchors.view(1, N, 4).expand(batch_size, N, 4).contiguous()
        gt_boxes = gt_boxes[:,:,:4].contiguous()


        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)
        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)

    elif anchors.dim() == 3:
        N = anchors.size(1)
        K = gt_boxes.size(1)

        if anchors.size(2) == 4:
            anchors = anchors[:,:,:4].contiguous()
        else:
            anchors = anchors[:,:,1:5].contiguous()

        gt_boxes = gt_boxes[:,:,:4].contiguous()

        gt_boxes_x = (gt_boxes[:,:,2] - gt_boxes[:,:,0] + 1)
        gt_boxes_y = (gt_boxes[:,:,3] - gt_boxes[:,:,1] + 1)
        gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

        anchors_boxes_x = (anchors[:,:,2] - anchors[:,:,0] + 1)
        anchors_boxes_y = (anchors[:,:,3] - anchors[:,:,1] + 1)
        anchors_area = (anchors_boxes_x * anchors_boxes_y).view(batch_size, N, 1)

        gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
        anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

        boxes = anchors.view(batch_size, N, 1, 4).expand(batch_size, N, K, 4)
        query_boxes = gt_boxes.view(batch_size, 1, K, 4).expand(batch_size, N, K, 4)

        iw = (torch.min(boxes[:,:,:,2], query_boxes[:,:,:,2]) -
            torch.max(boxes[:,:,:,0], query_boxes[:,:,:,0]) + 1)
        iw[iw < 0] = 0

        ih = (torch.min(boxes[:,:,:,3], query_boxes[:,:,:,3]) -
            torch.max(boxes[:,:,:,1], query_boxes[:,:,:,1]) + 1)
        ih[ih < 0] = 0
        ua = anchors_area + gt_boxes_area - (iw * ih)

        overlaps = iw * ih / ua

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(batch_size, N, 1).expand(batch_size, N, K), -1)
    else:
        raise ValueError('anchors input dimension is not correct.')

    return overlaps


class Interpolate(nn.Module):
    def __init__(self, size, mode= 'bilinear'):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.size = size
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, size=self.size, mode=self.mode, align_corners = True)
        return x

class ReferCam(nn.Module):
    def __init__(self, leaky=False):
        super(ReferCam, self).__init__()

        embin_size=512*2+8
        emb_size=256
        cam_size=20


        self.fcn_out = nn.Sequential(OrderedDict([
            ('0', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
                Interpolate(size=(cam_size, cam_size), mode='bilinear'),
                ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
            ('1', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
                Interpolate(size=(cam_size, cam_size), mode='bilinear'),
                ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
            ('2', torch.nn.Sequential(
                ConvBatchNormReLU(embin_size, emb_size, 1, 1, 0, 1, leaky=leaky),
                # ConvBatchNormReLU(emb_size*2, emb_size, 1, 1, 0, 1, leaky=leaky),
                Interpolate(size=(cam_size, cam_size), mode='bilinear'),
                ConvBatchNormReLU(emb_size, emb_size // 2, 1, 1, 0, 1, leaky=leaky),
                nn.Conv2d(emb_size // 2, 2, kernel_size=1), )),
        ]))

        self.avg_pool = nn.AvgPool2d(20)

    def forward(self, input):
        # args = get_args()

        if self.training:
            (intmd_fea, bbox, pred_anchor, args)  =input
            anchors_full = get_archors_full(args)
            batch_size=args.batch_size
            # n_neg=3
            roi_feat_all=[]
            scores=[]
            # iou_all=best_n_list
            roi_batch_all=[]
            label_batch_all=[]
            for scale_ii in range(len(pred_anchor)):

                grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
                anchor_idxs = [x + 3 * scale_ii for x in [0, 1, 2]]
                anchors = [anchors_full[i] for i in anchor_idxs]
                # scaled_anchors = torch.from_numpy(np.asarray([(x[0] / (args.anchor_imsize / grid), \
                #                    x[1] / (args.anchor_imsize / grid)) for x in anchors])).float()

                ws = np.asarray([np.round(x[0] * grid_size / (args.anchor_imsize / grid)) for x in anchors])
                hs = np.asarray([np.round(x[1] * grid_size / (args.anchor_imsize / grid)) for x in anchors])

                x_ctr, y_ctr = (grid_size - 1) * 0.5, (grid_size - 1) * 0.5

                scaled_anchors = torch.from_numpy(_mkanchors(ws, hs, x_ctr, y_ctr)).float().cuda()


                bbox_deltas = pred_anchor[scale_ii][:,:,:4,:,:]

                feat_height, feat_width = grid, grid
                shift_x = np.arange(0, feat_width) * grid_size
                shift_y = np.arange(0, feat_height) * grid_size
                shift_x, shift_y = np.meshgrid(shift_x, shift_y)
                shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                                     shift_x.ravel(), shift_y.ravel())).transpose())
                shifts = shifts.contiguous().type_as(bbox_deltas).float()

                A = 3
                K = shifts.size(0)

                # self._anchors = self._anchors.type_as(scores)
                # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
                anchors = scaled_anchors.view(1, A, 4) + shifts.view(K, 1, 4)
                anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)

                bbox_deltas = bbox_deltas.permute(0, 1, 3, 4, 2).contiguous()
                bbox_deltas = bbox_deltas.view(batch_size, -1, 4)

                proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size,grid_size) # xyxy

                proposals = clip_boxes(proposals, args.size, batch_size)

                gt_boxes = bbox.clone().unsqueeze(1) #xyxy

                # gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()
                # gt_boxes_append[:, :, 1:5] = gt_boxes[:, :, :4]

                # Include ground-truth boxes in the set of candidate rois
                all_rois = torch.cat([proposals, gt_boxes], 1)

                overlaps = bbox_overlaps_batch(all_rois, gt_boxes)

                max_overlaps, gt_assignment = torch.max(overlaps, 2)

                batch_size = overlaps.size(0)
                num_proposal = overlaps.size(1)
                num_boxes_per_img = overlaps.size(2)

                offset = torch.arange(0, batch_size) * gt_boxes.size(1)
                offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment

                labels = gt_boxes[:, :, 3]
                labels[:, :]=1.
                labels = labels.contiguous().view(-1)[offset.view(-1)] \
                    .view(batch_size, -1)
                # labels = torch.ones(batch_size,1).cuda()

                FG_THRESH=0.5
                BG_THRESH_HI = 0.5
                BG_THRESH_LO= 0.00
                fg_rois_per_image = 2
                rois_per_image=8
                # roi_size=(scale_ii+1)*7

                labels_batch = labels.new(batch_size, rois_per_image).zero_()
                rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()

                for i in range(batch_size):
                    fg_inds = torch.nonzero(max_overlaps[i] >= FG_THRESH).view(-1)
                    fg_num_rois = fg_inds.numel()

                    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
                    bg_inds = torch.nonzero((max_overlaps[i] < BG_THRESH_HI) &
                                            (max_overlaps[i] >= BG_THRESH_LO)).view(-1)
                    bg_num_rois = bg_inds.numel()

                    if fg_num_rois > 0 and bg_num_rois > 0:
                        # sampling fg
                        fg_rois_per_this_image = fg_rois_per_image#min(fg_rois_per_image, fg_num_rois)

                        # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                        # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                        # use numpy instead.
                        # rand_num = torch.randperm(fg_num_rois).long().cuda()
                        if fg_rois_per_image < fg_num_rois:
                            rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()
                            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                        else:
                            rand_num = torch.from_numpy(np.random.choice(fg_num_rois,fg_rois_per_image,replace=True)).type_as(gt_boxes).long()
                            fg_inds = fg_inds[rand_num]
                        # sampling bg
                        bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image

                        # Seems torch.rand has a bug, it will generate very large number and make an error.
                        # We use numpy rand instead.
                        # rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                        rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                        rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                        bg_inds = bg_inds[rand_num]

                    elif fg_num_rois > 0 and bg_num_rois == 0:
                        # sampling fg
                        # rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                        rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                        rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                        fg_inds = fg_inds[rand_num]
                        fg_rois_per_this_image = rois_per_image
                        bg_rois_per_this_image = 0
                    elif bg_num_rois > 0 and fg_num_rois == 0:
                        # sampling bg
                        # rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                        rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                        rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                        bg_inds = bg_inds[rand_num]
                        bg_rois_per_this_image = rois_per_image
                        fg_rois_per_this_image = 0
                    else:
                        raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

                        # The indices that we're selecting (both fg and bg)
                    keep_inds = torch.cat([fg_inds, bg_inds], 0)

                    # Select sampled values from various arrays:
                    labels_batch[i].copy_(labels[i][keep_inds])

                    # Clamp labels for the background RoIs to 0
                    if fg_rois_per_this_image < rois_per_image:
                        labels_batch[i][fg_rois_per_this_image:] = 0

                    rois_batch[i,:, 1:] = all_rois[i][keep_inds]
                    rois_batch[i, :, 0] = i
                roi_batch_all.append(rois_batch)
                label_batch_all.append(labels_batch)


                # num_images = 1
                # rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)
                # fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
                # fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image
                #
                # labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
                #     all_rois, gt_boxes, fg_rois_per_image,
                #     rois_per_image, self._num_classes)


                # box_coder(bbox, pred_anchor[scale_ii],anchors_full,scale_ii,args)
                # pred_coord_all = []
                # pred_score_all = []
                # # pred_coord_neg = torch.zeros(batch_size*3, 4)
                # pos_ind_batch=[]
                # neg_ind_batch = []
                # scaled_gt = coord_list[scale_ii]
                # pred_scale=pred_anchor[scale_ii]
                # for ii in range(batch_size):
                #
                #
                #     pos_anchor_ind = (iou_all[scale_ii] >= 0.5)[ii]
                #     neg_anchor_ind = (iou_all[scale_ii] < 0.5)[ii]
                #     # pos_ind_batch.append(pos_anchor_ind)
                #     # neg_ind_batch.append(neg_anchor_ind)
                #
                #     pred_coord = torch.zeros(n_neg+1, 4).cuda()
                #     pred_score= torch.zeros(n_neg+1).cuda()
                #     # best_scale_ii = best_n_list[ii]//3
                #     grid, grid_size = args.size//(32//(2**scale_ii)), 32//(2**scale_ii)
                #     anchor_idxs = [x + 3*scale_ii for x in [0,1,2]]
                #     anchors = [anchors_full[i] for i in anchor_idxs]
                #     scaled_anchors = [ (x[0] / (args.anchor_imsize/grid), \
                #         x[1] / (args.anchor_imsize/grid)) for x in anchors]
                #
                #     gi=scaled_gt[ii,0].long()
                #     gj=scaled_gt[ii,1].long()
                #
                #     ind=torch.zeros(3, grid, grid)
                #     ind[pos_anchor_ind,gj,gi] =1.
                #
                #
                #     # grid_range_x = list(range(0, grid))
                #     # grid_range_y = list(range(0, grid))
                #     # grid_range_x.remove(gi[ii])
                #     # grid_range_y.remove(gj[ii])
                #
                #
                #
                #     pred_coord[0,0] = F.sigmoid(pred_anchor[scale_ii][ii, best_n_list[ii]%3, 0, gj[ii], gi[ii]]) + gi[ii].float()
                #     pred_coord[0,1] = F.sigmoid(pred_anchor[scale_ii][ii, best_n_list[ii]%3, 1, gj[ii], gi[ii]]) + gj[ii].float()
                #     pred_coord[0,2] = torch.exp(pred_anchor[scale_ii][ii, best_n_list[ii]%3, 2, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][0]
                #     pred_coord[0,3] = torch.exp(pred_anchor[scale_ii][ii, best_n_list[ii]%3, 3, gj[ii], gi[ii]]) * scaled_anchors[best_n_list[ii]%3][1]
                #     pred_coord[0,:] = pred_coord[0,:] * grid_ratio
                #
                #     pred_score[0]=1
                #
                #     for jj in range(1,n_neg+1):
                #         coord_x = torch.tensor(random.choice(grid_range_x)).cuda()
                #         coord_y = torch.tensor(random.choice(grid_range_y)).cuda()
                #
                #         pred_coord[jj,0] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 0, coord_y, coord_x]) + coord_x.float()
                #         pred_coord[jj,1] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 1, coord_y, coord_x]) + coord_y.float()
                #         pred_coord[jj,2] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 2, coord_y, coord_x]) * scaled_anchors[best_n_list[ii]%3][0]
                #         pred_coord[jj,3] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 3, coord_y, coord_x]) * scaled_anchors[best_n_list[ii]%3][1]
                #         pred_coord[jj,:] = pred_coord[jj,:] * grid_ratio
                #
                #     # coord_x = torch.tensor(random.choice(grid_range_x)).cuda()
                #     # coord_y = torch.tensor(random.choice(grid_range_y)).cuda()
                #     #
                #     #
                #     # pred_coord[2,0] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 0, coord_y, coord_x]) + coord_x.float()
                #     # pred_coord[2,1] = F.sigmoid(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 1, coord_y, coord_x]) + coord_y.float()
                #     # pred_coord[2,2] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 2, coord_y, coord_x]) * scaled_anchors[best_n_list[ii]%3][0]
                #     # pred_coord[2,3] = torch.exp(pred_anchor[best_scale_ii][ii, best_n_list[ii]%3, 3, coord_y, coord_x]) * scaled_anchors[best_n_list[ii]%3][1]
                #     # pred_coord[2,:] = pred_coord[2,:] * grid_ratio
                #
                #     pred_coord = xywh2xyxy(pred_coord)
                #     pred_coord_all.append(pred_coord.cuda())
                #     pred_score_all.append(pred_score)

                # feat_map=intmd_fea[scale_ii]
                # roi_feat=roi_align(feat_map,rois_batch.view(-1, 5),[roi_size,roi_size], 1./grid)
                # roi_feat_all.append(roi_feat)
                # scores.append(labels_batch.view(-1))
                # pred_coord_all.append(pred_coord)

            roi_batch_all=torch.cat(roi_batch_all)
            label_batch_all=torch.cat(label_batch_all)


            for scale_ii in range(len(intmd_fea)):
                grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
                roi_size = (scale_ii + 1) * 7

                feat_map=intmd_fea[scale_ii]
                # roi_scale=torch.cat([roi_batch_all.view(-1, 5)[:,0].unsqueeze(1),roi_batch_all.view(-1, 5)[:,1:]/grid_size],dim=1)
                roi_feat=roi_align(feat_map,roi_batch_all.view(-1, 5),[roi_size,roi_size], 1./grid_size)
                roi_feat_all.append(roi_feat)
                scores.append(label_batch_all.view(-1))


            cam, bi_score = [], []
            for ii in range(len(roi_feat_all)):
                output=self.fcn_out._modules[str(ii)](roi_feat_all[ii])
                cam.append(output)
                bi_score.append(self.avg_pool(cam[ii]).squeeze())

            return cam, bi_score, scores
        else:
            (intmd_fea, seg_bbox, args)=input
            batch_size = seg_bbox.size(0)
            # feats = seg_bbox.unsqueeze(0)
            rois_batch = seg_bbox.new(batch_size, 5).zero_()
            for ii in range(batch_size):
                rois_batch[ii, 1:] = seg_bbox[ii]
                rois_batch[ii, 0] = ii

            roi_feat_all=[]
            for scale_ii in range(len(intmd_fea)):
                grid, grid_size = args.size // (32 // (2 ** scale_ii)), 32 // (2 ** scale_ii)
                roi_size = (scale_ii + 1) * 7
                # for ii in range(batch_size):
                 #[x.unsqueeze(0) for x in seg_bbox[scale_ii]]
                feat_map = intmd_fea[scale_ii]
                roi_feat = roi_align(feat_map, rois_batch, [roi_size, roi_size],1./grid_size)
                roi_feat_all.append(roi_feat)

            cam, bi_score = [], []
            for ii in range(len(roi_feat_all)):
                output=self.fcn_out._modules[str(ii)](roi_feat_all[ii])
                cam.append(output)
                bi_score.append(self.avg_pool(cam[ii]).squeeze())

            return cam, bi_score