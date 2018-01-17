# Chainer\_Mask\_R-CNN   
Chainer implementation of Mask R-CNN - the multi-task network for object detection, object classification, and instance segmentation.
(https://arxiv.org/abs/1703.06870)   
<a href="README_JP.md">日本語版 README</a>   
[DeNA Tech Blog(JP)](https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html)   

## Examples
<img src="imgs/demo.gif" width="400px"></img>
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
- [ ] Precision Evaluator
- [ ] Feature Pyramid Network
- [ ] Pose Estimation

## Prerequisite
- Download 'ResNet-50-model.caffemodel' from the "OneDrive download" of [ResNet pretrained models](https://github.com/KaimingHe/deep-residual-networks#models) 
for model initialization and place it in ~/.chainer/dataset/pfnet/chainer/models/

- COCO 2017 dataset :
the COCO dataset can be downloaded and unzipped by:
```
bash getData.sh
```   
Generate the list file by:   
```
python utils/makecocolist.py
```
Setup the COCO API:   
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../../
```


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
'--out', '-o', default='result',  help='output directory'   
'--seed', '-s', type=int, default=0   
'--roialign', action='store_true', default=True, help='True: ROIAlign, False: ROIpooling'
'--step_size', '-ss', type=int, default=400000  
'--lr_step', '-ls', type=int, default=480000    
'--lr_initialchange', '-li', type=int, default=800     
'--pretrained', '-p', type=str, default='imagenet'   
'--snapshot', type=int, default=4000   
'--resume', type=str   
'--iteration', '-i', type=int, default=800000   
'--roi_size', '-r', type=int, default=7, help='ROI size for mask head input'
'--gamma', type=float, default=1, help='mask loss balancing factor'   
```

note that we use a subdivision-based updater to enable training with large batch size.


## Demo
Segment the objects in the input image by executing:   
```
python demo.py --image <input image> --modelfile result/snapshot_model.npz 
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
