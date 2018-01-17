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
        #print(bottom_data.shape, bottom_rois.shape, self.outh)
        #e.g. (batch, channel, h, w)=(1, 512, 38, 53) (n_rois, )=(128, 5)
        channels, height, width = bottom_data.shape[1:]
        #print(bottom_rois[0], height, width, self.spatial_scale)
        n_rois = bottom_rois.shape[0]
        top_data = cuda.cupy.empty((n_rois, channels, self.outh,
                                    self.outw), dtype=numpy.float32)
        self.argmax_data = cuda.cupy.empty(top_data.shape, numpy.int32)
        #pooles_width(height)=7
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 bottom_data, float32 spatial_scale, int32 channels,
            int32 height, int32 width, int32 pooled_height, int32 pooled_width,
            raw float32 bottom_rois
            ''',
            'float32 top_data, int32 argmax_data',
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
            float roi_width = max(roi_end_w - roi_start_w + 1, 1.0);
            float roi_height = max(roi_end_h - roi_start_h + 1, 1.0);

            // float bin size 
            float bin_size_h = roi_height / static_cast<float>(pooled_height);
            float bin_size_w = roi_width / static_cast<float>(pooled_width);
            float maxval = 0;
            int maxidx = -1;
            
            for (int j = 0; j < 4; j++) {
                int i = j / 2;
                float val = 0;
                // ROIAlign using the center of the bin
                int hstart = static_cast<int>(floor((static_cast<float>(ph) + 0.25 + i * 0.5)
                                              * bin_size_h));
                int wstart = static_cast<int>(floor((static_cast<float>(pw) + 0.25 + (j % 2) * 0.5)
                                              * bin_size_w));
                int hend = static_cast<int>(ceil((static_cast<float>(ph) + 0.25 + i * 0.5)
                                            * bin_size_h)+1);
                int wend = static_cast<int>(ceil((static_cast<float>(pw) + 0.25 + (j % 2) * 0.5)
                                            * bin_size_w)+1);
                hstart = min(max(hstart + roi_start_h, 0.0), static_cast<float>(height));
                hend = min(max(hend + roi_start_h, 0.0),static_cast<float>(height));
                wstart = min(max(wstart + roi_start_w, 0.0), static_cast<float>(width));
                wend = min(max(wend + roi_start_w, 0.0), static_cast<float>(width));
                //compute the max value in the bin
                int data_offset = (roi_batch_ind * channels + c) * height * width;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                        int bottom_index = h * width + w;
                        val += bottom_data[data_offset + bottom_index];
                    }
                }
                if (maxval < val) {
                    maxval = val;
                    maxidx = hstart * width + wstart; 
                }
            }
            top_data = maxval / 4;
            argmax_data = maxidx;
            
            ''', 'roi_pooling_2d_fwd'
        )(bottom_data, self.spatial_scale, channels, height, width,
          self.outh, self.outw, bottom_rois, top_data,
          self.argmax_data)
        return top_data,

    def backward_gpu(self, inputs, gy):
        bottom_rois = inputs[1]
        channels, height, width = self._bottom_data_shape[1:]
        bottom_diff = cuda.cupy.zeros(self._bottom_data_shape, numpy.float32)
        cuda.cupy.ElementwiseKernel(
            '''
            raw float32 top_diff, raw int32 argmax_data, int32 num_rois,
            float32 spatial_scale, int32 channels, int32 height, int32 width,
            int32 pooled_height, int32 pooled_width, raw float32 bottom_rois
            ''',
            'float32 bottom_diff',
            '''
            int w = i % width;
            int h = (i / width) % height;
            int c = (i / (width * height)) % channels;
            int num = i / (width * height * channels);

            float gradient = 0;
            // Accumulate gradient over all ROIs that pooled this element
            for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
                // Skip if ROI's batch index doesn't match num
                if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
                    continue;
                }
                // same as forward
                float roi_start_w = bottom_rois[roi_n * 5 + 1] * spatial_scale;
                float roi_start_h = bottom_rois[roi_n * 5 + 2] * spatial_scale;
                float roi_end_w = bottom_rois[roi_n * 5 + 3] * spatial_scale;
                float roi_end_h = bottom_rois[roi_n * 5 + 4] * spatial_scale;

                // Skip if ROI doesn't include (h, w)
                const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                                     h >= roi_start_h && h <= roi_end_h);
                if (!in_roi) {
                    continue;
                }

                int offset = (roi_n * channels + c) * pooled_height * pooled_width;

                // Compute feasible set of pooled units that could have pooled
                // this bottom unit

                // Force malformed ROIs to be 1x1
                float roi_width = max(roi_end_w - roi_start_w + 1, 1.0);
                float roi_height = max(roi_end_h - roi_start_h + 1, 1.0);

                float bin_size_h = static_cast<float>(roi_height)
                               / static_cast<float>(pooled_height);
                float bin_size_w = static_cast<float>(roi_width)
                               / static_cast<float>(pooled_width);
                // ROIAlign using the center of the bin
                int phstart = floor(static_cast<float>(h - roi_start_h)
                                    / bin_size_h);
                int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                                 / bin_size_h);
                int pwstart = floor(static_cast<float>(w - roi_start_w)
                                    / bin_size_w);
                int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                                 / bin_size_w);

                phstart = min(max(phstart, 0), pooled_height);
                phend = min(max(phend, 0), pooled_height);
                pwstart = min(max(pwstart, 0), pooled_width);
                pwend = min(max(pwend, 0), pooled_width);
    
                for (int ph = phstart; ph < phend; ++ph) {
                    for (int pw = pwstart; pw < pwend; ++pw) {
                        int index_ = ph * pooled_width + pw + offset;
                        if (argmax_data[index_] == (h * width + w) ||
                        argmax_data[index_] == (h * width + w - 1) ||
                        argmax_data[index_] == ((h - 1) * width + w) ||
                        argmax_data[index_] == ((h - 1) * width + w - 1)) {
                            gradient += top_diff[index_];
                        }
                    }
                }
            }
            bottom_diff = gradient / 4;
            ''', 'roi_pooling_2d_bwd'
        )(gy[0], self.argmax_data, bottom_rois.shape[0], self.spatial_scale,
          channels, height, width, self.outh, self.outw,
          bottom_rois, bottom_diff)
        
        #bottom_diff.shape : e.g. (1,512, 38,50)

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
