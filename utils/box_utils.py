import numpy as np
import cupy
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

def bbox_yxyx2xywh(bbox):
    bbox_o = bbox.copy()
    bbox_o[:, 0] = bbox[:, 1]
    bbox_o[:, 2] = bbox[:, 3] - bbox[:, 1]
    bbox_o[:, 1] = bbox[:, 0]
    bbox_o[:, 3] = bbox[:, 2] - bbox[:, 0]
    return bbox_o

def im_mask(mask, size, bbox):
    # bboxes are already clipped to [0, w], [0, h]
    masksize = mask.shape[0]
    # pad the mask to avoid cv2.resize artifacts 
    pmask = np.zeros((masksize + 2, masksize + 2), dtype=np.float32)
    pmask[1:-1, 1:-1] = mask
    # extend the boxhead
    scale = (masksize + 2) / masksize
    ex_w = (bbox[3] - bbox[1]) * scale
    ex_h = (bbox[2] - bbox[0]) * scale
    ex_x0 = (bbox[3] + bbox[1] - ex_w) / 2
    ex_y0 = (bbox[2] + bbox[0] - ex_h) / 2
    ex_x1 = (bbox[3] + bbox[1] + ex_w) / 2
    ex_y1 = (bbox[2] + bbox[0] + ex_h) / 2
    ex_bbox = np.asarray([ex_y0, ex_x0, ex_y1, ex_x1], dtype=np.int32)
    # whole-image-sized mask 
    immask = np.zeros((size[0],size[1]), dtype=np.uint8)
    x0, x1 = max(ex_bbox[1], 0), min(ex_bbox[3] + 1, size[1])
    y0, y1= max(ex_bbox[0], 0), min(ex_bbox[2] + 1, size[0])
    immask_roi = cv2.resize(pmask, (x1 - x0, y1 - y0))
    immask[y0:y1, x0:x1] = np.round(immask_roi).astype(np.uint8)
    return immask
