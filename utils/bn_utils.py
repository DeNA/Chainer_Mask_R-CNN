import numpy as np
import cupy

def freeze_bn(model):
    # freeze batchnorm update 
    def disableupdate(block):
        for name in block._forward:
            l = getattr(block, name)
            l.bn1.disable_update()   
            l.bn2.disable_update()   
            l.bn3.disable_update()   
            if name=='a':
                l.bn4.disable_update()
    model.extractor.bn1.disable_update()  
    disableupdate(model.extractor.res2)
    disableupdate(model.extractor.res3)
    disableupdate(model.extractor.res4)
    disableupdate(model.head.res5)
    print("batchnorm update disabled!")

def bn_to_affine(model):
    # change batchnorm layers to affine layers (mean -> 0, var -> 1)
    def bn_to_affine_block(block):
        for name in block._forward:
            l = getattr(block, name)
            l.bn1.avg_mean = cupy.zeros(l.bn1.avg_mean.shape, dtype=np.float32)
            l.bn1.avg_var = cupy.ones(l.bn1.avg_var.shape, dtype=np.float32) - l.bn1.eps
            l.bn2.avg_mean = cupy.zeros(l.bn2.avg_mean.shape, dtype=np.float32)
            l.bn2.avg_var = cupy.ones(l.bn2.avg_var.shape, dtype=np.float32) - l.bn1.eps   
            l.bn3.avg_mean = cupy.zeros(l.bn3.avg_mean.shape, dtype=np.float32) 
            l.bn3.avg_var = cupy.ones(l.bn3.avg_var.shape, dtype=np.float32) - l.bn1.eps  
            if name=='a':
                l.bn4.avg_mean = cupy.zeros(l.bn4.avg_mean.shape, dtype=np.float32) 
                l.bn4.avg_var = cupy.ones(l.bn4.avg_var.shape, dtype=np.float32) - l.bn1.eps 
    model.extractor.bn1.avg_mean = cupy.zeros(model.extractor.bn1.avg_mean.shape, dtype=np.float32)
    model.extractor.bn1.avg_var = cupy.ones(model.extractor.bn1.avg_var.shape, dtype=np.float32) - model.extractor.bn1.eps 
    bn_to_affine_block(model.extractor.res2)
    bn_to_affine_block(model.extractor.res3)
    bn_to_affine_block(model.extractor.res4)
    bn_to_affine_block(model.head.res5)
    print("converted batchnorm to affine")