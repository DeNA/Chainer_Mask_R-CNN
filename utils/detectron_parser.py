import numpy as np
import os
path = os.path.join(os.path.dirname(__file__), '../')
import sys
sys.path.append(path)
from mask_rcnn_resnet import MaskRCNNResNet
from chainer import serializers
import pickle

model = MaskRCNNResNet(n_fg_class=80, roi_size=14, pretrained_model='auto', anchor_scales=[2, 4, 8, 16, 32], n_layers=50, class_ids=[[1]])

modeldir = "modelfiles"
if os.path.exists(modeldir)==False:
    os.mkdir(modeldir)
    
# resnet50, end-to-end, C4
d_model_file = "modelfiles/model_final.pkl"
c_model_file = "modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz"

with open(d_model_file, 'rb') as f:
    d = pickle.load(f, encoding='latin-1')['blobs']
d_key  = sorted(d)

parsecount = 0
for bl in d_key:
    if 'res' in bl:
        stage = bl[3] # resnet stage, 2, 3, 4, 5
        block = bl[5] # resnet block, a or b
        if stage=='_': # non-resnet layers
            continue
        else:
            stage = int(stage) - 1
            if stage == 4:
                netname='head'
            else:
                netname='extractor'
            if 'branch2a' in bl:
                c_nlayer = 1
            elif 'branch2b' in bl:
                c_nlayer = 2
            elif 'branch2c' in bl:
                c_nlayer = 3
            elif 'branch1' in bl:
                c_nlayer = 4
            else:
                c_nlayer = 0
            
            # do not copy
            if bl.endswith('_b') and 'bn_b' not in bl:
                continue
            if 'momentum' in bl:
                continue
            
            # conv / bn gamma / bn beta
            if '_w' in bl:
                c_kind = 'conv%d.W' % c_nlayer
            elif 'bn_s' in bl:
                c_kind = 'bn%d.gamma' % c_nlayer
            elif 'bn_b' in bl:
                c_kind = 'bn%d.beta' % c_nlayer
                
            # chainer block kind
            if block == '0':
                c_block = 'a'
            else:
                c_block = 'b'+block
            
            # shape checker
            exec("c_shape = model.%s.res%d.%s.%s.data.shape" % (netname, stage + 1, c_block, c_kind))
            exec("d_shape = d['%s'].shape" % bl)
            if c_shape == d_shape:
                # execute copy
                txt = "model.%s.res%d.%s.%s.data = d['%s']" % (netname, stage + 1, c_block, c_kind, bl )
                print(txt)
                exec(txt)
                parsecount += 1
            else:
                print("shape mismatch error!")

# copy the other layers
layer_pairs = \
[('extractor.conv1.W', 'conv1_w'), ('extractor.bn1.gamma', 'res_conv1_bn_s'), ('extractor.bn1.beta', 'res_conv1_bn_b'),
 ('rpn.conv1.W', 'conv_rpn_w'), ('rpn.conv1.b', 'conv_rpn_b'), 
 ('rpn.loc.W', 'rpn_bbox_pred_w'), ('rpn.loc.b', 'rpn_bbox_pred_b'), 
 ('rpn.score.W', 'rpn_cls_logits_w'), ('rpn.score.b', 'rpn_cls_logits_b'), 
 ('head.score.W', 'cls_score_w'), ('head.score.b', 'cls_score_b'), 
 ('head.cls_loc.W', 'bbox_pred_w'), ('head.cls_loc.b', 'bbox_pred_b'), 
 ('head.deconvm1.W', 'conv5_mask_w'), ('head.deconvm1.b', 'conv5_mask_b'),
 ('head.convm2.W', 'mask_fcn_logits_w'), ('head.convm2.b', 'mask_fcn_logits_b'),
]

def xytrans(src):
    sh = src.shape
    dst = src.reshape(sh[0]//4, 4, -1)[:,[1, 0, 3, 2]].reshape(sh)
    return dst

for layer_pair in layer_pairs:
    exec("c_shape = model.%s.data.shape" % layer_pair[0])
    exec("d_shape = d['%s'].shape" % layer_pair[1])
    if 'bbox_pred' in layer_pair[1]:
        d[layer_pair[1]] = xytrans(d[layer_pair[1]])
    if c_shape == d_shape:
        txt = "model.%s.data = d['%s']" % layer_pair
        print(txt)
        exec(txt)
        parsecount += 1
    else:
        print("shape mismatch error!")

print(parsecount, " layers copied")
serializers.save_npz(c_model_file, model)
print("save weights file to a chainer model", c_model_file)