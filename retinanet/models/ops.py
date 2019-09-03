# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import paddle.fluid as fluid
from paddle.fluid.dygraph.base import to_variable
import detectron.cython_bbox as cython_bbox
import detectron.cython_fgfake as cython_fgfake
bbox_overlaps = cython_bbox.bbox_overlaps
find_fg_fake_inds = cython_fgfake.find_fg_fake_inds

def _bbox_overlaps(roi_boxes, gt_boxes):
    w1 = np.maximum(roi_boxes[:, 2] - roi_boxes[:, 0] + 1, 0)
    h1 = np.maximum(roi_boxes[:, 3] - roi_boxes[:, 1] + 1, 0)
    w2 = np.maximum(gt_boxes[:, 2] - gt_boxes[:, 0] + 1, 0)
    h2 = np.maximum(gt_boxes[:, 3] - gt_boxes[:, 1] + 1, 0)
    area1 = w1 * h1
    area2 = w2 * h2

    overlaps = np.zeros((roi_boxes.shape[0], gt_boxes.shape[0]))
    for ind1 in range(roi_boxes.shape[0]):
        for ind2 in range(gt_boxes.shape[0]):
            inter_x1 = np.maximum(roi_boxes[ind1, 0], gt_boxes[ind2, 0])
            inter_y1 = np.maximum(roi_boxes[ind1, 1], gt_boxes[ind2, 1])
            inter_x2 = np.minimum(roi_boxes[ind1, 2], gt_boxes[ind2, 2])
            inter_y2 = np.minimum(roi_boxes[ind1, 3], gt_boxes[ind2, 3])
            inter_w = np.maximum(inter_x2 - inter_x1 + 1, 0)
            inter_h = np.maximum(inter_y2 - inter_y1 + 1, 0)
            inter_area = inter_w * inter_h
            iou = inter_area / (area1[ind1] + area2[ind2] - inter_area)
            overlaps[ind1, ind2] = iou
    return overlaps


def _box_to_delta(ex_boxes, gt_boxes, weights):
    ex_w = ex_boxes[:, 2] - ex_boxes[:, 0] + 1
    ex_h = ex_boxes[:, 3] - ex_boxes[:, 1] + 1
    ex_ctr_x = ex_boxes[:, 0] + 0.5 * ex_w
    ex_ctr_y = ex_boxes[:, 1] + 0.5 * ex_h

    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0] + 1
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1] + 1
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_w
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_h

    dx = (gt_ctr_x - ex_ctr_x) / ex_w / weights[0]
    dy = (gt_ctr_y - ex_ctr_y) / ex_h / weights[1]
    dw = (np.log(gt_w / ex_w)) / weights[2]
    dh = (np.log(gt_h / ex_h)) / weights[3]

    targets = np.vstack([dx, dy, dw, dh]).transpose()
    return targets

def _target_assign(anchor_by_gt_overlap, gt_labels, positive_overlap,
                            negative_overlap):
    anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
    anchor_to_gt_max = anchor_by_gt_overlap[np.arange(
        anchor_by_gt_overlap.shape[0]), anchor_to_gt_argmax]

    gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
    gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(
        anchor_by_gt_overlap.shape[1])]
    anchors_with_max_overlap = np.where(
        anchor_by_gt_overlap == gt_to_anchor_max)[0]

    labels = np.ones((anchor_by_gt_overlap.shape[0]), dtype=np.int32) * -1
    labels[anchors_with_max_overlap] = 1
    labels[anchor_to_gt_max >= positive_overlap] = 1

    fg_inds = np.where(labels == 1)[0]
    bbox_inside_weight = np.zeros((len(fg_inds), 4), dtype=np.float32)

    bg_inds = np.where(anchor_to_gt_max < negative_overlap)[0]
    enable_inds = bg_inds

    fg_fake_inds = np.array([], np.int32)
    fg_value = np.array([fg_inds[0]], np.int32)
    fake_num = 0
    #start_time = time.time()
    '''
    for bg_id in enable_inds:
        if bg_id in fg_inds:
            fake_num += 1
            fg_fake_inds = np.hstack([fg_fake_inds, fg_value])
    '''
    fg_fake_inds = find_fg_fake_inds(enable_inds, fg_inds, fg_inds[0])
    fake_num = len(fg_fake_inds)
    labels[enable_inds] = 0
    #end_time = time.time()
    #total_time = end_time - start_time
    #print('inner loop time: {}'.format(total_time))

    bbox_inside_weight[fake_num:, :] = 1
    fg_inds = np.where(labels == 1)[0]
    bg_inds = np.where(labels == 0)[0]
    loc_index = np.hstack([fg_fake_inds, fg_inds])
    score_index = np.hstack([fg_inds, bg_inds])
    score_index_tmp = np.hstack([fg_inds])
    labels = labels[score_index]

    gt_inds = anchor_to_gt_argmax[loc_index]
    label_inds = anchor_to_gt_argmax[score_index_tmp]
    labels[0:len(fg_inds)] = np.squeeze(gt_labels[label_inds])
    fg_num = len(fg_fake_inds) + len(fg_inds) + 1
    assert not np.any(labels == -1), "Wrong labels with -1"
    labels = labels[:, np.newaxis]
    return loc_index, score_index, labels, gt_inds, bbox_inside_weight, fg_num


def retinanet_target_assign(bbox_pred,
                            cls_logits,
                            anchor_box,
                            anchor_var,
                            gt_boxes,
                            gt_labels,
                            is_crowd,
                            im_info,
                            num_classes=1,
                            positive_overlap=0.5,
                            negative_overlap=0.4):
    anchor_box_data = anchor_box.numpy()
    anchor_num = anchor_box_data.shape[0]
    batch_size = gt_boxes.shape[0]

    for i in range(batch_size):
        #start_time = time.time()
        im_scale = im_info[i][2]

        inds_inside = np.arange(anchor_num)
        inside_anchors = anchor_box_data
        gt_boxes_slice = gt_boxes[i] * im_scale
        gt_labels_slice = gt_labels[i]
        is_crowd_slice = is_crowd[i]

        not_crowd_inds = np.where(is_crowd_slice == 0)[0]
        gt_boxes_slice = gt_boxes_slice[not_crowd_inds]
        gt_labels_slice = gt_labels_slice[not_crowd_inds]
        iou = bbox_overlaps(inside_anchors, gt_boxes_slice)

        loc_inds, score_inds, labels, gt_inds, bbox_inside_weight, fg_num_slice = \
                         _target_assign(iou, gt_labels_slice,
                                        positive_overlap, negative_overlap)
        # unmap to all anchor
        loc_inds = inds_inside[loc_inds]
        score_inds = inds_inside[score_inds]

        sampled_gt = gt_boxes_slice[gt_inds]
        sampled_anchor = anchor_box_data[loc_inds]
        box_deltas = _box_to_delta(sampled_anchor, sampled_gt, [1., 1., 1., 1.])

        if i == 0:
            loc_index_data = loc_inds
            score_index_data = score_inds
            target_label_data = labels
            target_bbox_data = box_deltas
            bbox_inside_weight_data = bbox_inside_weight
            fg_num_data = [fg_num_slice]
        else:
            loc_index_data = np.concatenate(
                [loc_index_data, loc_inds + i * anchor_num])
            score_index_data = np.concatenate(
                [score_index_data, score_inds + i * anchor_num])
            target_label_data = np.concatenate([target_label_data, labels])
            target_bbox_data = np.vstack([target_bbox_data, box_deltas])
            bbox_inside_weight_data = np.vstack([bbox_inside_weight_data, \
                                             bbox_inside_weight])
            fg_num_data = np.concatenate([fg_num_data, [fg_num_slice]])
        #end_time = time.time()
        #total_time = end_time - start_time
        #print('outer loop time: {}'.format(total_time))

    loc_index = to_variable(loc_index_data)
    score_index = to_variable(score_index_data)
    target_label = to_variable(target_label_data)
    target_bbox = to_variable(target_bbox_data)
    bbox_inside_weight = to_variable(bbox_inside_weight_data)
    fg_num = to_variable(np.array(fg_num_data).astype('int32'))

    loc_index._stop_gradient = True
    score_index._stop_gradient = True
    target_label._stop_gradient = True
    target_bbox._stop_gradient = True
    bbox_inside_weight._stop_gradient = True
    fg_num._stop_gradient = True

    cls_logits = fluid.layers.reshape(x=cls_logits, shape=(-1, num_classes))
    bbox_pred = fluid.layers.reshape(x=bbox_pred, shape=(-1, 4))
    predicted_cls_logits = fluid.layers.gather(cls_logits, score_index)
    predicted_bbox_pred = fluid.layers.gather(bbox_pred, loc_index)

    return predicted_cls_logits, predicted_bbox_pred, target_label, target_bbox, bbox_inside_weight, fg_num
