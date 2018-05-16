import numpy as np
from skimage.draw import polygon
import json
import os
import cv2
import pycocotools
from pycocotools.coco import COCO

import chainer
from chainercv.utils import read_image
class COCODataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir='COCO/', json_file='instances_train2017.json',
                 name='train2017', id_list_file='train2017.txt', sizemin=10):
        self.data_dir  = data_dir
        self.json_file = json_file
        self.coco = COCO(self.data_dir + 'annotations/'+self.json_file)
        self.ids = self.coco.getImgIds()
        self.name = name
        self.sizemin = sizemin
        self.class_ids = sorted(self.coco.getCatIds())

    def __len__(self):
        return len(self.ids)

    def ann2rle(self, ann, height, width):
        if isinstance(ann, list):
            rles = pycocotools.mask.frPyObjects(ann, height, width)
            rle = pycocotools.mask.merge(rles)
        elif isinstance(ann['counts'], list):
            rle = pycocotools.mask.frPyObjects(ann, height, width)
        else:
            rle = ann
        return rle

    def get_example(self, i):
        #i = i % 500 # for limiting data size
        numofboxes=0
        while True:
            id_ = self.ids[i]
            annot_labels, annot_bboxes, annot_segs= list(), list(), list()
            anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=None)
            annotations = self.coco.loadAnns(anno_ids)
            for a in annotations:
                if a['bbox'][2] > self.sizemin and a['bbox'][3] > self.sizemin \
                and a['iscrowd']==0:
                    annot_labels.append(a['category_id'])
                    annot_bboxes.append(a['bbox'])
                    annot_segs.append(a['segmentation'])
            numofboxes=len(annot_labels)
            if numofboxes > 0 or chainer.config.train == False:
                break
            else:
                i = i - 1
        img_file = os.path.join(self.data_dir, self.name, '{:012}'.format(id_) + '.jpg')
        img = read_image(img_file, color=True)
        _, h, w = img.shape
        annot_masks = []
        for annot_seg_polygons in annot_segs:
            rle = self.ann2rle(annot_seg_polygons, h, w)
            annot_masks.append(pycocotools.mask.decode(rle))
        if numofboxes > 0:
            annot_masks = np.stack(annot_masks).astype(np.uint8) #y,x
            annot_bboxes = np.stack(annot_bboxes).astype(np.float32)
            annot_labels = np.stack(annot_labels).astype(np.int32)
        else:
            annot_labels, annot_bboxes, annot_masks = [], [], []

        return img, annot_labels, annot_bboxes, annot_masks, i
