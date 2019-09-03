#  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License. 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from edict import AttrDict
import six
import numpy as np

_C = AttrDict()
cfg = _C

#
# Training options
#
_C.TRAIN = AttrDict()

# scales an image's shortest side
_C.TRAIN.scales = [800]

# max size of longest side
_C.TRAIN.max_size = 1333

# images per GPU in minibatch
_C.TRAIN.im_per_batch = 1 

# If False, only resize image and not pad, image shape is different between
# GPUs in one mini-batch. If True, image shape is the same in one mini-batch.
_C.TRAIN.padding_minibatch = False

# Snapshot period
_C.TRAIN.snapshot_iter = 40000


# remove anchors out of the image
_C.TRAIN.straddle_thresh = 0.


# min overlap between anchor and gt box to be a positive examples
_C.TRAIN.positive_overlap = 0.5

# max overlap between anchor and gt box to be a negative examples
_C.TRAIN.negative_overlap = 0.4

# stopgrad at a specified stage
_C.TRAIN.freeze_at = 2

# Use horizontally-flipped images during training?
_C.TRAIN.use_flipped = True

_C.TRAIN.gt_min_area = -1

# Focal losss
_C.TRAIN.gamma = 2.0

_C.TRAIN.alpha = 0.25

# Smooth L1 loss
_C.TRAIN.sigma = 3.0151134457776365
#
# Inference options
#
_C.TEST = AttrDict()

# scales an image's shortest side
_C.TEST.scales = [800]

# max size of longest side
_C.TEST.max_size = 1333

# min score threshold to infer
_C.TEST.score_thresh = 0.05

# overlap threshold used for NMS
_C.TEST.nms_thresh = 0.5

# Maximum number of detections to be kept according to the confidences 
# after the filtering detections based on score_threshold
_C.TEST.nms_top_k = 1000

# max number of detections
_C.TEST.detections_per_im = 100

_C.TEST.eta = 1.0


#
# Model options
#
_C.MASK_ON = False

_C.FPN_coarsest_stride = 128

_C.FPN_ON = True
# Anchor scales per octave
_C.scales_per_octave = 3

# At each FPN level, we generate anchors based on their scale, aspect_ratio,
# stride of the level, and we multiply the resulting anchor by anchor_scale
_C.anchor_scale = 4

# retinanet anchor ratiox
_C.aspect_ratio = [1.0, 2.0, 0.5]

# Convolutions to use in the cls and bbox tower
# NOTE: this doesn't include the last conv for logits
_C.num_convs = 4

_C.start_level = 0

_C.anchor_strides = [8, 16, 32, 64, 128]

# variance of anchors
_C.variances = [1., 1., 1., 1.]

# sampling ratio for roi align
_C.sampling_ratio = 0


# spatial scale 
_C.spatial_scale = 1. / 16.


#
# SOLVER options
#

# derived learning rate the to get the final learning rate.
_C.learning_rate = 0.00125

# maximum number of iterations, 1x: 180000, 2x:360000
_C.max_iter = 720000
#_C.max_iter = 360000

# warm up to learning rate 
_C.warm_up_iter = 4000
_C.warm_up_factor = 1. / 3

# lr steps_with_decay, 1x: [120000, 160000], 2x: [240000, 320000]
_C.lr_steps = [480000, 640000]
_C.lr_gamma = 0.1

# L2 regularization hyperparameter
_C.weight_decay = 0.0001

# momentum with SGD
_C.momentum = 0.9

#
# ENV options
#

# support both CPU and GPU
_C.use_gpu = True

# Whether use parallel
_C.parallel = True

# Class number
_C.class_num = 81

# support pyreader
_C.use_pyreader = True

# pixel mean values
_C.pixel_means = [102.9801, 115.9465, 122.7717]

# clip box to prevent overflowing
_C.bbox_clip = np.log(1000. / 16.)


def merge_cfg_from_args(args, mode):
    """Merge config keys, values in args into the global config."""
    if mode == 'train':
        sub_d = _C.TRAIN
    else:
        sub_d = _C.TEST
    for k, v in sorted(six.iteritems(vars(args))):
        d = _C
        try:
            value = eval(v)
        except:
            value = v
        if k in sub_d:
            sub_d[k] = value
        else:
            d[k] = value
