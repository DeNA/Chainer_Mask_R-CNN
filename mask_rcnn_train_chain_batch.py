import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F

from chainercv.links.model.faster_rcnn.utils.anchor_target_creator import AnchorTargetCreator
from utils.proposal_target_creator import ProposalTargetCreator
from chainer import computational_graph as c
from chainercv.links import PixelwiseSoftmaxClassifier

class MaskRCNNTrainChain(chainer.Chain):
    def __init__(self, mask_rcnn, rpn_sigma=3., roi_sigma=1., gamma=1,
                 anchor_target_creator=AnchorTargetCreator(),
                 roi_size=7):
        super(MaskRCNNTrainChain, self).__init__()
        with self.init_scope():
            self.mask_rcnn = mask_rcnn
        self.rpn_sigma = rpn_sigma
        self.roi_sigma = roi_sigma
        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = ProposalTargetCreator(roi_size=roi_size)
        self.loc_normalize_mean = mask_rcnn.loc_normalize_mean
        self.loc_normalize_std = mask_rcnn.loc_normalize_std
        self.decayrate=0.99
        self.avg_loss = None
        self.gamma=gamma
    def __call__(self, imgs, bboxes, labels, scale, masks):

        if isinstance(bboxes, chainer.Variable):
            bboxes = bboxes.data
        if isinstance(labels, chainer.Variable):
            labels = labels.data
        if isinstance(scale, chainer.Variable):
            scale = scale.data
        if isinstance(masks, chainer.Variable):
            masks = masks.data
        scale = np.asscalar(cuda.to_cpu(scale[0]))
        n = bboxes.shape[0]
        #if n != 1:
        #    raise ValueError('only batch size 1 is supported')
        _, _, H, W = imgs.shape
        img_size = (H, W)
        #Extractor (VGG) : img -> features
        features = self.mask_rcnn.extractor(imgs)

        #Region Proposal Network : features -> rpn_locs, rpn_scores, rois
        rpn_loc_loss,rpn_cls_loss, roi_loc_loss, roi_cls_loss, mask_loss= 0,0,0,0,0    
        for i in range(n):
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.mask_rcnn.rpn(
                features[i:i+1], img_size, scale)
            bbox, label, mask, rpn_score, rpn_loc, roi = \
                bboxes[i], labels[i], masks[i], rpn_scores[0], rpn_locs[0], rois
            mask[mask>1]=0
            numdata = sum(label>=0)
            label = label[0:numdata]
            bbox = bbox[0:numdata]
            mask = mask[0:numdata]
            #proposal target : roi(proposed) , bbox(GT), label(GT) -> sample_roi, gt_roi_loc, gt_roi_label
            #the targets are compared with the head output.
            sample_roi, gt_roi_loc, gt_roi_label, gt_roi_mask = self.proposal_target_creator(
            roi, bbox, label, mask, self.loc_normalize_mean, self.loc_normalize_std)
            sample_roi_index = self.xp.zeros((len(sample_roi),), dtype=np.int32)

            #Head Network : features, sample_roi -> roi_cls_loc, roi_score
            roi_cls_loc, roi_score, roi_cls_mask = self.mask_rcnn.head(
                features[i:i+1], sample_roi, sample_roi_index)

            #RPN losses
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(bbox, anchor, img_size)
            rpn_loc_loss += _fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label, self.rpn_sigma)
            rpn_cls_loss += F.softmax_cross_entropy(rpn_score, gt_rpn_label)

            #Head output losses
            n_sample = roi_cls_loc.shape[0]
            roi_cls_loc = roi_cls_loc.reshape((n_sample, -1, 4))
            roi_loc = roi_cls_loc[self.xp.arange(n_sample), gt_roi_label] 
            roi_mask = roi_cls_mask[self.xp.arange(n_sample), gt_roi_label]
            roi_loc_loss += _fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label, self.roi_sigma)
            roi_cls_loss += F.softmax_cross_entropy(roi_score, gt_roi_label)

            #mask loss:  average binary cross-entropy loss
            mask_loss += F.sigmoid_cross_entropy(roi_mask[0:gt_roi_mask.shape[0]], gt_roi_mask)

        #total loss
        loss = rpn_loc_loss + rpn_cls_loss + roi_loc_loss + roi_cls_loss + self.gamma * mask_loss
        loss /= n

        #avg loss calculation
        if self.avg_loss is None:
            self.avg_loss = loss.data
        else:
            self.avg_loss = self.avg_loss * self.decayrate + loss.data*(1-self.decayrate)
        chainer.reporter.report({'rpn_loc_loss':rpn_loc_loss/n,
                                 'rpn_cls_loss':rpn_cls_loss/n,
                                 'roi_loc_loss':roi_loc_loss/n,
                                 'roi_cls_loss':roi_cls_loss/n,
                                 'roi_mask_loss':self.gamma * mask_loss/n,
                                 'avg_loss':self.avg_loss,
                                 'loss':loss}, self)
        return loss


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = F.absolute(diff)
    flag = (abs_diff.data < (1. / sigma2)).astype(np.float32)
    y = (flag * (sigma2 / 2.) * F.square(diff) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return F.sum(y)

def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    xp = chainer.cuda.get_array_module(pred_loc)
    in_weight = xp.zeros_like(gt_loc)
    in_weight[gt_label > 0] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight, sigma)
    loc_loss /= xp.sum(gt_label >= 0)
    return loc_loss
