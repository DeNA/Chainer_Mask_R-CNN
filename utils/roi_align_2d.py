# Modified work as ROIAlign:
# -----------------------------------------------------------------------------
# Copyright (c) 2018 DeNA
# -----------------------------------------------------------------------------

# Modified work:
# -----------------------------------------------------------------------------
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
# -----------------------------------------------------------------------------

# Original work of forward_gpu and backward_gpu:
# -----------------------------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see fast-rcnn/LICENSE for details]
# Written by Ross Girshick
# -----------------------------------------------------------------------------

import numpy
import six

from chainer import cuda
from chainer import function
from chainer.utils import type_check

class ROIAlign2D(function.Function):

    """RoI align over a set of 2d planes."""

    def __init__(self, outh, outw, spatial_scale):
        self.outh, self.outw = outh, outw
        self.spatial_scale = spatial_scale

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)

        x_type, roi_type = in_types
        type_check.expect(
            x_type.dtype == numpy.float32,
            x_type.ndim == 4,
            roi_type.dtype == numpy.float32,
            roi_type.ndim == 2,
            roi_type.shape[1] == 5,
        )

    def forward_gpu(self, inputs):
        self.retain_inputs((1,))
        self._bottom_data_shape = inputs[0].shape

        bottom_data, bottom_rois = inputs
        #e.g. (batch, channel, h, w)=(1, 512, 38, 53) (n_rois, )=(128, 5)
        channels, height, width = bottom_data.shape[1:]
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            // scale the ROI coordinates (1/16)
            float roi_batch_ind = bottom_rois[num * 5 + 0];
            float roi_start_w = bottom_rois[num * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[num * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[num * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[num * 5 + 4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = max(roi_end_w - roi_start_w, 1.0);
            float roi_height = max(roi_end_h - roi_start_h, 1.0);

            // float bin size 
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);
            float maxval = 0;
            int maxidx = -1;
            
            for (int j = 0; j < 4; j++) {
                int ih = j / 2;
                int iw = j % 2;
                float val = 0;
                // ROIAlign using the center of the bin
                float fh = roi_start_h + (static_cast<float>(ph) + 0.25 + static_cast<float>(ih) * 0.5f) * bin_size_h;
                float fw = roi_start_w + (static_cast<float>(pw) + 0.25 + static_cast<float>(iw) * 0.5f) * bin_size_w;
                
                if (fh < -1.0 || fh > height || fw < -1.0 || fw > width) {
                    continue;
                }

                int hstart = static_cast<int>(floor(fh));
                int wstart = static_cast<int>(floor(fw));
                int hend = hstart + 1;
                int wend = wstart + 1;

                if (hstart >= height - 1) {
                    hend = hstart = height - 1;
                    fh = static_cast<float>(hstart);
                } else {
                    hend = hstart + 1;
                }

                if (wstart >= width - 1) {
                    wend = wstart = width - 1;
                    fw = static_cast<float>(wstart);
                } else {
                    wend = wstart + 1;
                }
                float dh = fh - static_cast<float>(hstart);
                float dw = fw - static_cast<float>(wstart);

                //compute the max value in the bin
                int data_offset = (roi_batch_ind * channels + c) * height * width;

                val += (1.0 - dh) * (1.0 - dw) * bottom_data[data_offset + hstart * width + wstart];
                val += (1.0 - dh) * dw         * bottom_data[data_offset + hstart * width + wend];
                val += dh * (1.0 - dw)         * bottom_data[data_offset + hend * width + wstart];
                val += dh * dw                 * bottom_data[data_offset + hend * width + wend];

                maxval += val;
            }
            top_data = maxval / 4;
            
            ''', 'roi_pooling_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data)
        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''',
            'raw float32 bottom_diff',
            '''
            // pos in output filter
            int pw = i % pooled_width;
            int ph = (i / pooled_width) % pooled_height;
            int c = (i / pooled_width / pooled_height) % channels;
            int num = i / pooled_width / pooled_height / channels;

            // scale the ROI coordinates (1/16)
            float roi_batch_ind = bottom_rois[num * 5 + 0];
            float roi_start_w = bottom_rois[num * 5 + 1] * spatial_scale;
            float roi_start_h = bottom_rois[num * 5 + 2] * spatial_scale;
            float roi_end_w = bottom_rois[num * 5 + 3] * spatial_scale;
            float roi_end_h = bottom_rois[num * 5 + 4] * spatial_scale;

            // Force malformed ROIs to be 1x1
            float roi_width = max(roi_end_w - roi_start_w, 1.0);
            float roi_height = max(roi_end_h - roi_start_h, 1.0);

            // float bin size 
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);
            int data_offset = (roi_batch_ind * channels + c) * height * width;
            
            for (int j = 0; j < 4; j++) {
                int ih = j / 2;
                int iw = j % 2;
                // ROIAlign using the center of the bin
                float fh = roi_start_h + (static_cast<float>(ph) + 0.25 + static_cast<float>(ih) * 0.5f) * bin_size_h;
                float fw = roi_start_w + (static_cast<float>(pw) + 0.25 + static_cast<float>(iw) * 0.5f) * bin_size_w;
                
                if (fh < -1.0 || fh > height || fw < -1.0 || fw > width) {
                    continue;
                }

                int hstart = static_cast<int>(floor(fh));
                int wstart = static_cast<int>(floor(fw));
                int hend = hstart + 1;
                int wend = wstart + 1;

                if (hstart >= height - 1) {
                    hend = hstart = height - 1;
                    fh = static_cast<float>(hstart);
                } else {
                    hend = hstart + 1;
                }

                if (wstart >= width - 1) {
                    wend = wstart = width - 1;
                    fw = static_cast<float>(wstart);
                } else {
                    wend = wstart + 1;
                }
                float dh = fh - static_cast<float>(hstart);
                float dw = fw - static_cast<float>(wstart);

                //atomic add: pointer, value
                atomicAdd(&bottom_diff[data_offset + hstart * width + wstart], top_diff[i] * (1.0 - dh) * (1.0 - dw) / 4);
                atomicAdd(&bottom_diff[data_offset + hstart * width + wend], top_diff[i] * (1.0 - dh) * dw         / 4);
                atomicAdd(&bottom_diff[data_offset + hend * width + wstart], top_diff[i] * dh         * (1.0 - dw) / 4);
                atomicAdd(&bottom_diff[data_offset + hend * width + wend], top_diff[i] * dh         * dw         / 4);
            }

            ''', 'roi_pooling_2d_bwd'
        )(gy[0], bottom_rois.shape[0], self.spatial_scale,
          channels, height, width, self.outh, self.outw,
          bottom_rois, bottom_diff, size=gy[0].size)
        
        return bottom_diff, None


def roi_align_2d(x, rois, outh, outw, spatial_scale):
    """Spatial Region of Interest (ROI) align function.

    This function acts similarly to :class:`~functions.MaxPooling2D`, but
    it computes the maximum of input spatial patch for each channel
    with the region of interest.

    Args:
        x (~chainer.Variable): Input variable. The shape is expected to be
            4 dimentional: (n: batch, c: channel, h, height, w: width).
        rois (~chainer.Variable): Input roi variable. The shape is expected to
            be (n: data size, 5), and each datum is set as below:
            (batch_index, x_min, y_min, x_max, y_max).
        outh (int): Height of output image after pooled.
        outw (int): Width of output image after pooled.
        spatial_scale (float): Scale of the roi is resized.

    Returns:
        ~chainer.Variable: Output variable.

    See the original paper proposing ROIPooling:
    `Fast R-CNN <https://arxiv.org/abs/1504.08083>`_.

    """
    return ROIAlign2D(outh, outw, spatial_scale)(x, rois)
