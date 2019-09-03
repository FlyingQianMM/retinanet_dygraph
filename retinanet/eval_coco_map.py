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
from models.model_builder import *
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval, Params
from config import cfg
from data_utils import DatasetPath

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def eval():

    data_path = DatasetPath('val')
    test_list = data_path.get_file_list()

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

    retinanet = RetinaNet("retinanet")

    if cfg.pretrained_model:
        pretrained, _ = fluid.dygraph.load_persistables(cfg.pretrained_model)
        retinanet.load_dict(pretrained)

    # yapf: enable
    test_reader = reader.test(total_batch_size)

    dts_res = []
    segms_res = []
    
    eval_start = time.time()
    retinanet.eval()
    for batch_id, batch_data in enumerate(test_reader()):
        start = time.time()
        im_info = []
        for data in batch_data:
            im_info.append(data[1])
        image_data = np.array(
            [x[0] for x in batch_data]).astype('float32')
        im_info_data = np.array(
            [x[1] for x in batch_data]).astype('float32')
        im_id_data = np.array(
            [x[2] for x in batch_data]).astype('int32')

        pred_boxes = retinanet('eval', image_data, im_info_data)
        nmsed_out = pred_boxes['det_outs']
        lod = pred_boxes['lod']

        dts_res += get_dt_res(total_batch_size, lod, nmsed_out,
                              batch_data, num_id_to_cat_id_map)

        end = time.time()
        print('batch id: {}, time: {}'.format(batch_id, end - start))
        #break
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
    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        eval()
