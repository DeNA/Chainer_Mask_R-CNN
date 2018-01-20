# Chainer\_Mask\_R-CNN   
マルチタスク検出器Mask R-CNNのchainer実装
(https://arxiv.org/abs/1703.06870)   

[DeNA Tech Blogでの解説](https://engineer.dena.jp/2017/12/chainercvmask-r-cnn.html)

## 実行例
<img src="imgs/demo.gif" width="400px"></img>
## 必要環境
- [Chainer](https://github.com/pfnet/chainer)
- [Chainercv](https://github.com/chainer/chainercv)
- [Cupy](https://github.com/cupy/cupy)   
 (動作確認済み: chainer==3.1.0, chainercv==0.7.0, verified: cupy==1.0.3)
```
$ pip install chainer   
$ pip install chainercv
$ pip install cupy==1.0.3
```   
- Python 3.0+   
- NumPy   
- Matplotlib   
- OpenCV   

## TODOs
- [ ] Evaluator
- [ ] Feature Pyramid Network
- [ ] Pose Estimation

## ファイル準備
- 学習済みモデルのダウンロード  
・以下リンク先の'OneDrive download'から、ResNet-50-model.caffemodelをダウンロード
 [ResNet pretrained models](https://github.com/KaimingHe/deep-residual-networks#models)
・~/.chainer/dataset/pfnet/chainer/models/　に置く

- COCO 2017 データセット
COCOデータセットのダウンロードと解凍:   
```
bash getData.sh
```
リストファイルの生成
```
python utils/makecocolist.py
```
- COCO APIのセットアップ:   
```
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI/
make
python setup.py install
cd ../../
```


## 学習

```
python train.py 
```
引数は以下です:
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

本実装ではsubdivisionを用いたupdateを行なっているため、batch size = 1 相当のGPUメモリでbatch size=8等を指定可能です

## デモ
入力画像のインスタンス・セグメンテーションを実行します:   
```
python demo.py --image <input image> --modelfile result/snapshot_model.npz 
```

## 引用
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
