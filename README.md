# Chainer\_Mask\_R-CNN   
Chainer implementation of Mask R-CNN - the multi-task network for object detection, object classification, and instance segmentation.
(https://arxiv.org/abs/1703.06870)   
<a href="README_JP.md">日本語版 README</a>   

## What's New

- Training result for R-50-C4 model has been evaluated!
- COCO box AP = 0.346 using our trainer (0.355 with official boxes) 
- COCO mask AP = 0.287 using our trainer (0.314 with official boxes) 

## Examples
- to be updated

## Requirements
- [Chainer](https://github.com/pfnet/chainer)
- [Chainercv](https://github.com/chainer/chainercv)
- [Cupy](https://github.com/cupy/cupy)   
(operable if your environment can run chainer > v3 with cuda and cudnn.)   
(verified as operable: chainer==3.1.0, chainercv==0.7.0, cupy==1.0.3)
```
$ pip install chainer   
$ pip install chainercv
$ pip install cupy
```   
- Python 3.0+   
- NumPy   
- Matplotlib   
- OpenCV   

## TODOs
- [x] Precision Evaluator (bbox, COCO metric)
- [x] Detectron Model Parser 
- [x] Modify ROIAlign
- [x] Mask inference using refined ROIs
- [x] Precision Evaluator (mask, COCO metric)
- [ ] Improve segmentation AP for R-50-C4 model
- [ ] Feature Pyramid Network (R-50-FPN)
- [ ] Keypoint Detection (R-50-FPN, Keypoints)

## Benchmark Results

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> Box AP 50:95</td><td bgcolor=white> Segm AP 50:95</td></tr>
<tr><th align="left" bgcolor=#f8f8f8>Ours (1 GPU)</th> <td bgcolor=white> 0.346 </td><td bgcolor=white> 0.287 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8>Detectron model</th> <td bgcolor=white> 0.350 </td><td bgcolor=white> 0.295 </td></tr>
<tr><th align="left" bgcolor=#f8f8f8>Detectron caffe2</th> <td bgcolor=white> 0.355 </td><td bgcolor=white> 0.314 </td></tr>
</table></tbody>

## Inference with Pretrained Models

- Download the pretrained model from the [Model Zoo] (https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)   
 (`model` link of `R-50-C4	Mask` at `End-to-End Faster & Mask R-CNN Baselines`)   
- Make `modelfiles` directory and put the downloaded file `model_final.pkl` in it   
- Execute:  
```   
python utils/detectron_parser.py
```
- And the converted model file is saved in `modelfiles`
- Run the demo:
```
python demo.py --bn2affine --modelfile modelfiles/e2e_mask_rcnn_R-50-C4_1x_d2c.npz --image <input image>
```

## Prerequisites for training
- Download 'ResNet-50-model.caffemodel' from the "OneDrive download" of [ResNet pretrained models](https://github.com/KaimingHe/deep-residual-networks#models) 
for model initialization and place it in ~/.chainer/dataset/pfnet/chainer/models/

- COCO 2017 dataset :
the COCO dataset can be downloaded and unzipped by:
```
bash getcoco.sh
```   
Setup the COCO API:   
```
git clone https://github.com/waleedka/coco
cd coco/PythonAPI/
make
python setup.py install
cd ../../
```
note: the official coco repository is not python3 compatible.    
Use the repository above in order to run our evaluation.    

## Train

```
python train.py 
```
arguments and the default conditions are defined as follows:
```
'--dataset', choices=('coco2017'), default='coco2017'   
'--extractor', choices=('resnet50','resnet101'), default='resnet50', help='extractor network'
'--gpu', '-g', type=int, default=0   
'--lr', '-l', type=float, default=1e-4   
'--batchsize', '-b', type=int, default=8   
'--freeze_bn', action='store_true', default=False, help='freeze batchnorm gamma/beta'
'--bn2affine', action='store_true', default=False, help='batchnorm to affine'
'--out', '-o', default='result',  help='output directory'   
'--seed', '-s', type=int, default=0   
'--roialign', action='store_true', default=True, help='True: ROIAlign, False: ROIpooling'
'--step_size', '-ss', type=int, default=400000  
'--lr_step', '-ls', type=int, default=480000    
'--lr_initialchange', '-li', type=int, default=800     
'--pretrained', '-p', type=str, default='imagenet'   
'--snapshot', type=int, default=4000   
'--validation', type=int, default=30000   
'--resume', type=str   
'--iteration', '-i', type=int, default=800000   
'--roi_size', '-r', type=int, default=14, help='ROI size for mask head input'
'--gamma', type=float, default=1, help='mask loss balancing factor'   
```

note that we use a subdivision-based updater to enable training with large batch size.


## Demo
Segment the objects in the input image by executing:   
```
python demo.py --image <input image> --modelfile result/snapshot_model.npz --contour
```

## Evaluation

Evaluate the trained model with COCO metric (bounding box, segmentation) :   
```
python train.py --lr 0 --iteration 1 --validation 1 --resume <trained_model> 
```

## Citation
Please cite the original paper in your publications if it helps your research:    

    @article{DBLP:journals/corr/HeGDG17,
      author    = {Kaiming He and
                  Georgia Gkioxari and
                  Piotr Doll{\'{a}}r and
                  Ross B. Girshick},
      title     = {Mask {R-CNN}},
      journal   = {CoRR},
      volume    = {abs/1703.06870},
      year      = {2017},
      url       = {http://arxiv.org/abs/1703.06870},
      archivePrefix = {arXiv},
      eprint    = {1703.06870},
      timestamp = {Wed, 07 Jun 2017 14:42:32 +0200},
      biburl    = {http://dblp.org/rec/bib/journals/corr/HeGDG17},
      bibsource = {dblp computer science bibliography, http://dblp.org}
    }
