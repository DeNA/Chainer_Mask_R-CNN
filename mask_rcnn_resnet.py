import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L
from mask_rcnn import MaskRCNN
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from utils import roi_align_2d
from chainer.links.model.vision.resnet import BuildingBlock, ResNetLayers

class ExtractorResNet(ResNetLayers):
    def __init__(self, pretrained_model='auto', n_layers=50):
        print('ResNet',n_layers,' initialization')
        if pretrained_model=='auto':
            if n_layers == 50:
                pretrained_model = 'ResNet-50-model.caffemodel'
            elif n_layers == 101:
                pretrained_model = 'ResNet-101-model.caffemodel'
        super(ExtractorResNet, self).__init__(pretrained_model, n_layers)
        del self.fc6
        del self.res5
    def __call__(self, x):
        h = self.conv1(x)
        h = F.max_pooling_2d(h, ksize=3, stride=2)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        return h

class MaskRCNNResNet(MaskRCNN):
    feat_stride = 16
    def __init__(self,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5 ,1, 2], anchor_scales=[8, 16, 32],
                 initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params=dict(),
                 roi_size=7,
                 n_layers=50, 
                 roi_align=True
                 ):
        print("MaskRNNResNet initialization")
        if n_fg_class is None:
            raise ValueError('supply n_fg_class!')
        if loc_initialW is None:
            loc_initialW = chainer.initializers.Normal(0.001)
        if score_initialW is None:
            score_initialW = chainer.initializers.Normal(0.01)
        if rpn_initialW is None:
            rpn_initialW = chainer.initializers.Normal(0.01)
        if initialW is None:# and pretrained_model:
            print("setting initialW")
            initialW = chainer.initializers.Normal(0.01)
        self.roi_size=roi_size
        if pretrained_model is not None:
            pretrained_model = 'auto'
        extractor = ExtractorResNet(pretrained_model, n_layers=n_layers)
        rpn = RegionProposalNetwork(
            1024, 1024,
            ratios=ratios, anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )
        head = MaskRCNNHead(
            n_fg_class + 1,
            roi_size=self.roi_size, spatial_scale=1. / self.feat_stride,
            initialW=initialW, loc_initialW=loc_initialW, score_initialW=score_initialW,
            roi_align=roi_align
        )
        super(MaskRCNNResNet, self).__init__(
            extractor, rpn, head,
            mean=np.array([122.7717, 115.9465, 102.9801], dtype=np.float32)[:, None, None],
            min_size=min_size, max_size=max_size
        )

class MaskRCNNHead(chainer.Chain):
    def __init__(self, n_class, roi_size, spatial_scale,
                 initialW=None, loc_initialW=None, score_initialW=None, roi_align=True):
        super(MaskRCNNHead, self).__init__()
        with self.init_scope():
            self.res5 = BuildingBlock(3, 1024, 512, 2048, 1, initialW=initialW) 
            #class / loc branch
            self.cls_loc = L.Linear(2048, n_class * 4, initialW=initialW)
            self.score = L.Linear(2048, n_class, initialW=score_initialW)
            #Mask-RCNN branch
            self.deconvm1 = L.Deconvolution2D(2048, 256, 2, 2, initialW=initialW)
            self.convm2 = L.Convolution2D(256, n_class, 3, 1, pad=1,initialW=initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.roi_align = roi_align
        print("ROI Align=",roi_align)

    def __call__(self, x, rois, roi_indices):
        roi_indices = roi_indices.astype(np.float32)
        indices_and_rois = self.xp.concatenate(
            (roi_indices[:, None], rois), axis=1)
        #x: (batch, channel, w, h)
        #rois: (128, 4) (ROI indices)
        if self.roi_align:
            pool = _roi_align_2d_yx(
                x, indices_and_rois, self.roi_size,self.roi_size,
                self.spatial_scale)
        else:
            pool = _roi_pooling_2d_yx(
                x, indices_and_rois, self.roi_size,self.roi_size,
                self.spatial_scale)

        #ROI, CLS  branch
        hres5 = self.res5(pool)
        fmap_size = hres5.shape[2:]
        h = F.average_pooling_2d(hres5, fmap_size, stride=1)
        roi_cls_locs = self.cls_loc(h)
        roi_scores = self.score(h)

        #Mask-RCNN branch
        h = F.relu(self.deconvm1(hres5)) 
        masks=self.convm2(h)
        return roi_cls_locs, roi_scores, masks

def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool

def _roi_align_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = roi_align_2d.roi_align_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
