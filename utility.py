#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
"""
Contains common utility functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import distutils.util
import numpy as np
import six
import collections
from collections import deque
import datetime
from paddle.fluid import core
import argparse
import functools
from config import *


def print_arguments(args):
    """Print argparse's arguments.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        parser.add_argument("name", default="Jonh", type=str, help="User name.")
        args = parser.parse_args()
        print_arguments(args)

    :param args: Input argparse.Namespace for printing.
    :type args: argparse.Namespace
    """
    print("-----------  Configuration Arguments -----------")
    for arg, value in sorted(six.iteritems(vars(args))):
        print("%s: %s" % (arg, value))
    print("------------------------------------------------")


def add_arguments(argname, type, default, help, argparser, **kwargs):
    """Add argparse's argument.

    Usage:

    .. code-block:: python

        parser = argparse.ArgumentParser()
        add_argument("name", str, "Jonh", "User name.", parser)
        args = parser.parse_args()
    """
    type = distutils.util.strtobool if type == bool else type
    argparser.add_argument(
        "--" + argname,
        default=default,
        type=type,
        help=help + ' Default: %(default)s.',
        **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)

    def add_value(self, value):
        self.deque.append(value)

    def get_median_value(self):
        return np.median(self.deque)


def now_time():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')


class TrainingStats(object):
    def __init__(self, window_size, stats_keys):
        self.smoothed_losses_and_metrics = {
            key: SmoothedValue(window_size)
            for key in stats_keys
        }

    def update(self, stats):
        for k, v in self.smoothed_losses_and_metrics.items():
            v.add_value(stats[k])

    def get(self, extras=None):
        stats = collections.OrderedDict()
        if extras:
            for k, v in extras.items():
                stats[k] = v
        for k, v in self.smoothed_losses_and_metrics.items():
            stats[k] = round(v.get_median_value(), 3)

        return stats

    def log(self, extras=None):
        d = self.get(extras)
        strs = ', '.join(str(dict({x: y})).strip('{}') for x, y in d.items())
        return strs


def parse_args():
    """return all args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    # ENV
    add_arg('use_data_parallel',         bool,   True,       "Whether use parallel.")
    add_arg('use_gpu',          bool,  True,      "Whether use GPU.")
    add_arg('model_save_dir',   str,    'output',     "The path to save model.")
    add_arg('pretrained_model', str,    'imagenet_resnet50_fusebn', "The init model path.")
    add_arg('dataset',          str,   'coco2017',  "coco2014, coco2017.")
    add_arg('class_num',        int,   81,          "Class number.")
    add_arg('data_dir',         str,   'dataset/coco',        "The data root path.")
    add_arg('padding_minibatch',bool,   False,
        "If False, only resize image and not pad, image shape is different between"
        " GPUs in one mini-batch. If True, image shape is the same in one mini-batch.")
    add_arg('num_gpus',         int,    1,          "GPG number.")
    #SOLVER
    add_arg('learning_rate',    float,  0.01,     "Learning rate.")
    add_arg('max_iter',         int,    90000,   "Iter number.")
    add_arg('lr_steps',         int,   [60000, 80000], "lr steps_with_decay.")
    add_arg('warm_up_iter',     int,    500,       "warm up to learning rate.")
    add_arg('log_window',       int,    20,        "Log smooth window, set 1 for debug, set 20 for train.")
    # TRAIN VAL INFER
    add_arg('im_per_batch',       int,   2,        "Minibatch size.")
    add_arg('max_size',         int,   1333,    "The resized image height.")
    add_arg('scales', int,  [800],    "The resized image height.")
    add_arg('batch_size_per_im',int,    512,    "fast rcnn head batch size")
    add_arg('pixel_means',     float,   [102.9801, 115.9465, 122.7717], "pixel mean")
    add_arg('snapshot_iter',  int,    10000,    "save model every snapshot stride.")
    add_arg('use_flipped',     bool,    True, "Use horizontally-flipped images during training?")
    # SINGLE EVAL AND DRAW
    add_arg('draw_threshold',  float, 0.8,    "Confidence threshold to draw bbox.")
    add_arg('image_path',       str,   'dataset/coco/val2017',  "The image path used to inference and visualize.")
    # ce
    add_arg('enable_ce',       bool,  False, 'If set, run the task with continuous evaluation logs.')
    # yapf: enable
    args = parser.parse_args()
    file_name = sys.argv[0]
    if 'train' in file_name or 'profile' in file_name:
        merge_cfg_from_args(args, 'train')
    else:
        merge_cfg_from_args(args, 'val')
    return args
