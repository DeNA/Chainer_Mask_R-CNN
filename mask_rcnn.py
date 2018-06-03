from __future__ import division

import numpy as np

import chainer
from chainer import cuda
import chainer.functions as F
from chainercv.links.model.faster_rcnn.utils.loc2bbox import loc2bbox
from chainercv.utils import non_maximum_suppression
from chainercv.transforms.image.resize import resize
import cv2
import pycocotools
from utils.box_utils import bbox_yxyx2xywh, im_mask

class MaskRCNN(chainer.Chain):
    def __init__(self, extractor, rpn, head, mean,
                 min_size=600, max_size=1000,
                 loc_normalize_mean=(0., 0., 0., 0.),
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2),
                 class_ids=[]
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
        if class_ids==[]:
            raise ValueError('set class ids')
        self.class_ids = class_ids
        self.preset = 'visualize'
    @property
    def n_class(self):
        return self.head.n_class

    def __call__(self, x, scale=1.):
        img_size = x.shape[2:]
        h = self.extractor(x) #VGG
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.rpn(h, img_size, scale) #Region Proposal Network
        hres5 = self.head.res5head(h, rois, roi_indices)
        roi_cls_locs, roi_scores = self.head.boxhead(hres5)
        return roi_cls_locs, roi_scores, rois, roi_indices, h

    def use_preset(self, preset):
        if preset == 'visualize':
            self.nms_thresh = 0.3
            self.score_thresh = 0.7
            self.preset = 'visualize'
        elif preset == 'evaluate':
            self.nms_thresh = 0.5
            self.score_thresh = 0.05
            self.preset = 'evaluate'
        else:
            raise ValueError('preset must be visualize or evaluate')

    def prepare(self, img):
        _, H, W = img.shape
        scale = self.min_size / min(H, W)
        if scale * max(H, W) > self.max_size:
            scale = self.max_size / max(H, W)
        #img = resize(img, (int(H * scale), int(W * scale)))
        img = img.transpose((1,2,0))
        img = cv2.resize(img, None, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_LINEAR)
        img = img.transpose((2,0,1))
        img = (img - self.mean).astype(np.float32, copy=False)
        img = img[::-1, :, :] # RGB to BGR order for resnet pretrained model
        return img

    def _suppress(self, raw_cls_bbox, raw_cls_roi, raw_prob):
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
            keep = non_maximum_suppression(cls_bbox_l, self.nms_thresh, prob_l)
            bbox.append(cls_bbox_l[keep])
            roi.append(cls_roi_l[keep])
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        roi = np.concatenate(roi, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.float32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, roi, label, score

    def predict(self, imgs):
        prepared_imgs = list()
        sizes = list()
        #print("predicting!")
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
                roi_cls_locs, roi_scores, rois, _,  h = self.__call__(img_var, scale=scale)
            #assuming batch size = 1
            roi_cls_loc = roi_cls_locs.data
            roi_score = roi_scores.data
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
            bbox, out_roi, label, score = self._suppress(raw_cls_bbox, raw_cls_roi, raw_prob)
            mask=[]
            if len(bbox) > 0:
                # mask head
                roi_indices = self.xp.zeros((len(bbox),), dtype=np.int32)
                with chainer.using_config('train', False), \
                    chainer.function.no_backprop_mode():
                    hres5 = self.head.res5head(h, cuda.to_gpu(bbox * scale), roi_indices)
                    roi_masks = self.head.maskhead(hres5)
                roi_mask = F.sigmoid(roi_masks).data
                raw_mask = cuda.to_cpu(roi_mask)
                # postprocess 
                if self.preset == 'evaluate':
                    bboxes.append(bbox_yxyx2xywh(bbox))
                    wmasks = []
                    for m, b, l in zip(raw_mask, bbox, label):
                        wm = im_mask(m[int(l+1)], size, b)
                        # encode the mask 
                        wm = pycocotools.mask.encode(np.asfortranarray(wm))
                        wm['counts'] = wm['counts'].decode('ascii')
                        mask.append(wm)
                elif self.preset == 'visualize':
                    bboxes.append(bbox)
                    wmasks = []
                    for m, b, l in zip(raw_mask, bbox, label):
                        wm = im_mask(m[int(l+1)], size, b)
                        mask.append(wm)
            elif self.preset == 'evaluate':
                # len(bbox) = 0
                wm = np.zeros((size[0], size[1]), dtype=np.uint8)
                wm = pycocotools.mask.encode(np.asfortranarray(wm))
                wm['counts'] = wm['counts'].decode('ascii')
                mask.append(wm)
                bboxes.append(bbox_yxyx2xywh(bbox))
            labels.append([self.class_ids[int(l)] for l in label.tolist()])
            scores.append(score)
            masks.append(mask)

        return bboxes, labels, scores, masks

