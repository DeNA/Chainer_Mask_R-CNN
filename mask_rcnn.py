from __future__ import division

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
from chainercv.transforms.image.resize import resize

class MaskRCNN(chainer.Chain):
    def __init__(self, extractor, rpn, head, mean,
                 min_size=600, max_size=1000,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 ):
        print("MaskRCNN initialization")
        super(MaskRCNN, self).__init__()
        with self.init_scope():
            self.extractor = extractor
            self.rpn = rpn
            self.head = head

        self.mean = mean
        self.min_size = min_size
        self.max_size = max_size
        self.loc_normalize_mean = loc_normalize_mean
        self.loc_normalize_std = loc_normalize_std
        self.use_preset('visualize')
    @property
    def n_class(self):
        return self.head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x) #VGG
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale) #Region Proposal Network
        roi_cls_locs, roi_scores, masks = self.head(
            h, rois, roi_indices) #Heads
        return roi_cls_locs, roi_scores, rois, roi_indices, masks

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
        elif preset == 'evaluate':
            self.nms_thresh = 0.3
            self.score_thresh = 0.05
        else:
            raise ValueError('preset must be visualize or evaluate')

    def prepare(self, img):
        _, H, W = img.shape
        scale = self.min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        img = resize(img, (int(H * scale), int(W * scale)))
        img = (img - self.mean).astype(np.float32, copy=False)
        return img

    def _suppress(self, raw_cls_bbox, raw_cls_roi, raw_prob, raw_mask):
        bbox = list()
        roi = list()
        label = list()
        score = list()
        mask = list()
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            cls_roi_l = raw_cls_roi.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            lmask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[lmask]
            cls_roi_l = cls_roi_l[lmask]
            prob_l = prob_l[lmask]
            mask_l = raw_mask[:,l]
            mask_l = mask_l[lmask]
            keep = non_maximum_suppression(cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            roi.append(cls_roi_l[keep])
            #labels are in [0, self.nclass - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
            mask.append(mask_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        roi = np.concatenate(roi, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.float32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        mask = np.concatenate(mask, axis=0).astype(np.float32)
        return bbox, roi,  label, score, mask

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        print("predicting!")
        for img in imgs:
            size = img.shape[1:]
            img = self.prepare(img.astype(np.float32))
            prepared_imgs.append(img)
            sizes.append(size)
        bboxes = list()
        out_rois = list()
        labels = list()
        scores = list()
        masks = list()
        for img, size in zip(prepared_imgs, sizes):
            with chainer.using_config('train', False), \
                chainer.function.no_backprop_mode():
                img_var = chainer.Variable(self.xp.asarray(img[None]))
                scale = img_var.shape[3] / size[1]
                roi_cls_locs, roi_scores, rois, _,  roi_masks = self.__call__(img_var, scale=scale)
           
            #assuming batch size = 1
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
            roi_mask = F.sigmoid(roi_masks).data
            roi = rois / scale
            mean = self.xp.tile(self.xp.asarray(self.loc_normalize_mean), self.n_class)
            std = self.xp.tile(self.xp.asarray(self.loc_normalize_std), self.n_class)
            roi_cls_loc = (roi_cls_loc * std + mean).astype(np.float32)
            roi_cls_loc = roi_cls_loc.reshape((-1, self.n_class, 4))
            roi = self.xp.broadcast_to(roi[:, None], roi_cls_loc.shape).reshape((-1, 4))
            cls_bbox = loc2bbox(roi, roi_cls_loc.reshape((-1, 4)))
            cls_bbox = cls_bbox.reshape((-1, self.n_class * 4))
            cls_roi = roi.reshape((-1, self.n_class * 4))
            #clip the bbox
            cls_bbox[:, 0::2] = self.xp.clip(cls_bbox[:, 0::2], 0, size[0])
            cls_bbox[:, 1::2] = self.xp.clip(cls_bbox[:, 1::2], 0, size[1])
            cls_roi[:, 0::2] = self.xp.clip(cls_roi[:, 0::2], 0, size[0])
            cls_roi[:, 1::2] = self.xp.clip(cls_roi[:, 1::2], 0, size[1])

            prob = F.softmax(roi_score).data

            raw_cls_bbox = cuda.to_cpu(cls_bbox)
            raw_cls_roi = cuda.to_cpu(cls_roi)
            raw_prob = cuda.to_cpu(prob)
            raw_mask = cuda.to_cpu(roi_mask)
            bbox, out_roi, label, score, mask = self._suppress(raw_cls_bbox, raw_cls_roi, raw_prob, raw_mask)
            bboxes.append(bbox)
            out_rois.append(out_roi)
            labels.append(label)
            scores.append(score)
            masks.append(mask)

        return bboxes, out_rois, labels, scores, masks

