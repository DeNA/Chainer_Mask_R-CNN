import chainer
from chainer import training
from chainer.training import extensions, ParallelUpdater
from chainer.training.triggers import ManualScheduleTrigger
from chainer.datasets import TransformDataset
from chainercv.datasets import VOCBboxDataset, voc_bbox_label_names
from chainercv import transforms
from chainercv.transforms.image.resize import resize

import argparse
import numpy as np
import time
#from mask_rcnn_vgg import MaskRCNNVGG16
from mask_rcnn_resnet import MaskRCNNResNet
from coco_dataset import COCODataset
from mask_rcnn_train_chain import MaskRCNNTrainChain
from utils.bn_utils import freeze_bn, bn_to_affine
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.detection_coco_evaluator import DetectionCOCOEvaluator
import logging
import traceback
from utils.updater import SubDivisionUpdater
import cv2

def resize_bbox(bbox, in_size, out_size):
    bbox_o = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox_o[:, 0] = y_scale * bbox[:, 1]
    bbox_o[:, 2] = y_scale * (bbox[:, 1]+bbox[:, 3])
    bbox_o[:, 1] = x_scale * bbox[:, 0]
    bbox_o[:, 3] = x_scale * (bbox[:, 0]+bbox[:, 2])
    return bbox_o

def parse():
    parser = argparse.ArgumentParser(
        description='Mask RCNN trainer')
    parser.add_argument('--dataset', choices=('coco2017'),
                        default='coco2017')
    parser.add_argument('--extractor', choices=('resnet50','resnet101'),
                        default='resnet50', help='extractor network')
    parser.add_argument('--gpu', '-g', type=int, default=0)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--batchsize', '-b', type=int, default=8)
    parser.add_argument('--freeze_bn', action='store_true', default=False, help='freeze batchnorm gamma/beta')
    parser.add_argument('--bn2affine', action='store_true', default=False, help='batchnorm to affine')
    parser.add_argument('--out', '-o', default='result',
                        help='Output directory')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--roialign', action='store_false', default=True, help='default: True')
    parser.add_argument('--lr_step', '-ls', type=int, default=120000)
    parser.add_argument('--lr_initialchange', '-li', type=int, default=400)
    parser.add_argument('--pretrained', '-p', type=str, default='imagenet')
    parser.add_argument('--snapshot', type=int, default=4000)
    parser.add_argument('--validation', type=int, default=30000)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--iteration', '-i', type=int, default=180000)
    parser.add_argument('--roi_size', '-r', type=int, default=14, help='ROI size for mask head input')
    parser.add_argument('--gamma', type=float, default=1, help='mask loss weight')
    return parser.parse_args()

class Transform(object):
    def __init__(self, net, labelids):
        self.net = net
        self.labelids = labelids
    def __call__(self, in_data):
        if len(in_data)==5:
            img, label, bbox, mask, i = in_data
        elif len(in_data)==4:
            img, bbox, label, i= in_data
        label = [self.labelids.index(l) for l in label]
        _, H, W = img.shape
        if chainer.config.train:
            img = self.net.prepare(img)
        _, o_H, o_W = img.shape
        scale = o_H / H
        if len(bbox)==0:
            return img, [],[],1
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))
        mask = resize(mask,(o_H, o_W))
        if chainer.config.train:
            #horizontal flip
            img, params = transforms.random_flip(
                img, x_random=True, return_param=True)
            bbox = transforms.flip_bbox(
                bbox, (o_H, o_W), x_flip=params['x_flip'])
            mask = transforms.flip(mask, x_flip=params['x_flip'])
        return img, bbox, label, scale, mask, i

def convert(batch, device):
    return chainer.dataset.convert.concat_examples(batch, device, padding=-1)

def main():
    args = parse()
    np.random.seed(args.seed)
    print('arguments: ', args)

    # Model setup
    if args.dataset == 'coco2017':
        train_data = COCODataset()
    test_data = COCODataset(json_file='instances_val2017.json', name='val2017', id_list_file='val2017.txt')
    train_class_ids =train_data.class_ids
    test_ids = test_data.ids
    cocoanns = test_data.coco
    if args.extractor=='vgg16':
        mask_rcnn = MaskRCNNVGG16(n_fg_class=80, pretrained_model=args.pretrained, roi_size=args.roi_size, roi_align = args.roialign)
    elif args.extractor=='resnet50':
        mask_rcnn = MaskRCNNResNet(n_fg_class=80, pretrained_model=args.pretrained,roi_size=args.roi_size, n_layers=50, roi_align = args.roialign, class_ids=train_class_ids)
    elif args.extractor=='resnet101':
        mask_rcnn = MaskRCNNResNet(n_fg_class=80, pretrained_model=args.pretrained,roi_size=args.roi_size, n_layers=101, roi_align = args.roialign, class_ids=train_class_ids)
    mask_rcnn.use_preset('evaluate')
    model = MaskRCNNTrainChain(mask_rcnn, gamma=args.gamma, roi_size=args.roi_size)
 
    # Trainer setup
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()
    optimizer = chainer.optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    #optimizer = chainer.optimizers.Adam()#alpha=0.001, beta1=0.9, beta2=0.999 , eps=0.00000001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=0.0001))

    train_data=TransformDataset(train_data, Transform(mask_rcnn, train_class_ids))
    test_data=TransformDataset(test_data, Transform(mask_rcnn, train_class_ids))
    train_iter = chainer.iterators.SerialIterator(
        train_data, batch_size=args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size=1, repeat=False, shuffle=False)
    updater = SubDivisionUpdater(train_iter, optimizer, device=args.gpu, subdivisions=args.batchsize)
    #updater = ParallelUpdater(train_iter, optimizer, devices={"main": 0, "second": 1}, converter=convert ) #for training with multiple GPUs
    trainer = training.Trainer(
        updater, (args.iteration, 'iteration'), out=args.out)

    # Extensions
    trainer.extend(
        extensions.snapshot_object(model.mask_rcnn, 'snapshot_model.npz'),
        trigger=(args.snapshot, 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 10),
                       trigger=ManualScheduleTrigger(
                          [args.lr_initialchange], 'iteration'))
    trainer.extend(extensions.ExponentialShift('lr', 0.1),
                   trigger=(args.lr_step, 'iteration'))
    if args.resume is not None:
        chainer.serializers.load_npz(args.resume, model.mask_rcnn)
    if args.freeze_bn:
        freeze_bn(model.mask_rcnn)
    if args.bn2affine:
        bn_to_affine(model.mask_rcnn)
    log_interval = 40, 'iteration'
    plot_interval = 160, 'iteration'
    print_interval = 40, 'iteration'

    #trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), trigger=(args.validation, 'iteration'))
    #trainer.extend(DetectionCOCOEvaluator(test_iter, model.mask_rcnn), trigger=(args.validation, 'iteration')) #COCO AP Evaluator with VOC metric
    trainer.extend(COCOAPIEvaluator(test_iter, model.mask_rcnn, test_ids, cocoanns), trigger=(args.validation, 'iteration')) #COCO AP Evaluator
    trainer.extend(chainer.training.extensions.observe_lr(),
                   trigger=log_interval)
    trainer.extend(extensions.LogReport(trigger=log_interval))
    trainer.extend(extensions.PrintReport(
        ['iteration', 'epoch', 'elapsed_time', 'lr',
         'main/loss',
         'main/avg_loss',
         'main/roi_loc_loss',
         'main/roi_cls_loss',
         'main/roi_mask_loss',
         'main/rpn_loc_loss',
         'main/rpn_cls_loss',
         'validation/main/loss',
         'validation/main/map',
         ]), trigger=print_interval)
    trainer.extend(extensions.ProgressBar(update_interval=1000))
    #trainer.extend(extensions.dump_graph('main/loss'))
    try:
        trainer.run()
    except:
        traceback.print_exc()

if __name__ == '__main__':
    main()
