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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np
from eval_helper import *
import paddle
import paddle.fluid as fluid
import reader
from utility import print_arguments, parse_args
import models.model_builder as model_builder
import models.resnet as resnet
import models.fpn as fpn
from  test_retinanet import detect_bbox
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from data_utils import DatasetPath

import errno
import hashlib
import logging
import os
import re
import six
import sys
from six.moves import cPickle as pickle
from six.moves import urllib
from uuid import uuid4
import numpy

def eval():

    data_path = DatasetPath('val')
    test_list = data_path.get_file_list()

    image_shape = [3, cfg.TEST.max_size, cfg.TEST.max_size]
    class_nums = cfg.class_num
    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch
    cocoGt = COCO(test_list)
    num_id_to_cat_id_map = {i + 1: v for i, v in enumerate(cocoGt.getCatIds())}
    category_ids = cocoGt.getCatIds()
    label_list = {
        item['id']: item['name']
        for item in cocoGt.loadCats(category_ids)
    }
    label_list[0] = ['background']

    model = model_builder.RetinaNet(add_conv_body_func=resnet.add_ResNet50_conv_body,
            add_fpn_neck_func=fpn.add_fpn_neck,
            anchor_strides=[8, 16, 32, 64, 128],
            use_pyreader=False,
            mode='val')
    model.build_model(image_shape)
    pred_boxes, pred_scores, im_info = model.eval_bbox_out()
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    # yapf: disable
    if cfg.pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
        ''' 
        w_sum = 0.
        cnt = 0 
        all_vars = fluid.default_main_program().global_block().vars
        for k, v in all_vars.items():
            if 'double' in k: continue
            if v.persistable and ('weights' in k or 'biases' in k or 'scale' in k or 'offset' in k or '_w' in k or '_b' in k) and ('tmp' not in k) and ('velocity' not in k):
                #bn_vars.append(k)
                #print(k)
                #w_sum += np.sum(np.array(fluid.global_scope().find_var(k).get_tensor()))
                cnt += 1
        print('weight sum ', cnt)
        print('==========================')
        # detectron
        params_d = np.load(os.path.join('detectron-model', 'model_final.pkl'))
        params = params_d['blobs']

        temp = set()
        cnt = 0
        for var in params.keys():
            var_name = var
            post = var[-2:]
            pre = var[:3]
            if pre != 'fpn':
                if 'conv1' in var and 'res' not in var:
                    if '_w' in var:
                        var_name = 'conv1_weights'
                elif 'conv1' in var:
                    if '_b' in post:
                        var_name='bn_conv1_offset'
                    elif '_s' in post:
                        var_name='bn_conv1_scale'
                    else:
                        var_name='conv1_weights'
                elif 'res' in var_name:
                    var_name = var[:4]+chr(ord("a")+int(var[5]))+var[6:]
                    if '_w' in post:
                        var_name = var_name[:-2]+'_weights'
                    if '_b' in post:
                        var_name = 'bn'+var_name[3:-5]+'_offset'
                    if '_s' in post:
                        var_name = 'bn'+var_name[3:-5]+'_scale'
    
            if os.path.exists('output_duiqi/init_model/'+var_name) and var_name not in temp:
                cnt += 1
                #print(var_name)
                temp.add(var_name)
                param = fluid.global_scope().find_var(var_name).get_tensor()
                if var == 'bbox_pred_w' or var=='cls_score_w' or 'fc' in var:
                    params[var] = np.transpose(params[var])
                param.set(params[var],place)

        print('load weight sum: ',cnt)
        '''
    # yapf: enable
    test_reader = reader.test(total_batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    dts_res = []
    segms_res = []
    
    fetch_list = [pred_boxes, pred_scores, im_info]
    eval_start = time.time()
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
        im_info = []
        for data in batch_data:
            im_info.append(data[1])
        print("begin run")
        results = exe.run(fetch_list=[v.name for v in fetch_list[0]] + [v.name for v in fetch_list[1]] + [fetch_list[2].name],
                          feed=feeder.feed(batch_data),
                          return_numpy=False)
        print("after run")

        pred_boxes_v = results[0:5]
        pred_scores_v = results[5:10]
        im_info_v = results[10]
        
        dts_res += detect_bbox(pred_boxes_v, pred_scores_v, im_info_v, batch_data, num_id_to_cat_id_map)
        
        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
    eval_end = time.time()
    total_time = eval_end - eval_start
    print('average time of eval is: {}'.format(total_time / (batch_id + 1)))
    assert len(dts_res) > 0, "The number of valid bbox detected is zero.\n \
        Please use reasonable model and check input data."

    with open("detection_bbox_result.json", 'w') as outfile:
        json.dump(dts_res, outfile)
    print("start evaluate bbox using coco api")
    cocoDt = cocoGt.loadRes("detection_bbox_result.json")
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    

if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    eval()
