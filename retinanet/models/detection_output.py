#  Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
#Licensed under the Apache License, Version 2.0 (the "License")
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

import numpy as np
import math
import copy


def iou(box_a, box_b, norm):
    """Apply intersection-over-union overlap between box_a and box_b
    """
    xmin_a = min(box_a[0], box_a[2])
    ymin_a = min(box_a[1], box_a[3])
    xmax_a = max(box_a[0], box_a[2])
    ymax_a = max(box_a[1], box_a[3])

    xmin_b = min(box_b[0], box_b[2])
    ymin_b = min(box_b[1], box_b[3])
    xmax_b = max(box_b[0], box_b[2])
    ymax_b = max(box_b[1], box_b[3])

    area_a = (ymax_a - ymin_a + (norm == False)) * (xmax_a - xmin_a +
                                                    (norm == False))
    area_b = (ymax_b - ymin_b + (norm == False)) * (xmax_b - xmin_b +
                                                    (norm == False))
    if area_a <= 0 and area_b <= 0:
        return 0.0

    xa = max(xmin_a, xmin_b)
    ya = max(ymin_a, ymin_b)
    xb = min(xmax_a, xmax_b)
    yb = min(ymax_a, ymax_b)

    inter_area = max(xb - xa + (norm == False),
                     0.0) * max(yb - ya + (norm == False), 0.0)

    iou_ratio = inter_area / (area_a + area_b - inter_area)

    return iou_ratio


def nms(boxes,
        scores,
        score_threshold,
        nms_threshold,
        normalized=True):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        score_threshold: (float) The confidence thresh for filtering low
            confidence boxes.
        nms_threshold: (float) The overlap thresh for suppressing unnecessary
            boxes.
        top_k: (int) The maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    all_scores = copy.deepcopy(scores)
    all_scores = all_scores.flatten()
    selected_indices = np.argwhere(all_scores > score_threshold)
    selected_indices = selected_indices.flatten()
    all_scores = all_scores[selected_indices]

    sorted_indices = np.argsort(-all_scores, axis=0, kind='mergesort')
    sorted_scores = all_scores[sorted_indices]
    sorted_indices = selected_indices[sorted_indices]

    selected_indices = []
    for i in range(sorted_scores.shape[0]):
        idx = sorted_indices[i]
        keep = True
        for k in range(len(selected_indices)):
            if keep:
                kept_idx = selected_indices[k]
                overlap = iou(boxes[idx], boxes[kept_idx], normalized)
                keep = True if overlap <= nms_threshold else False
            else:
                break
        if keep:
            selected_indices.append(idx)
    return selected_indices


def multiclass_nms(prediction, class_num, keep_top_k, nms_threshold):
    selected_indices = {}
    num_det = 0
    for c in range(class_num):
        if c not in prediction.keys():
            continue
        cls_dets = prediction[c]
        all_scores = np.zeros(len(cls_dets))
        for i in range(all_scores.shape[0]):
            all_scores[i] = cls_dets[i][4]
        indices = nms(cls_dets, all_scores, 0.0, nms_threshold, False)
        selected_indices[c] = indices
        num_det += len(indices)

    score_index = []
    for c, indices in selected_indices.items():
        for idx in indices:
            score_index.append((prediction[c][idx][4], c, idx))

    sorted_score_index = sorted(
        score_index, key=lambda tup: tup[0], reverse=True)
    if keep_top_k > -1 and num_det > keep_top_k:
        sorted_score_index = sorted_score_index[:keep_top_k]
        num_det = keep_top_k
    nmsed_outs = []
    for s, c, idx in sorted_score_index:
        xmin = prediction[c][idx][0]
        ymin = prediction[c][idx][1]
        xmax = prediction[c][idx][2]
        ymax = prediction[c][idx][3]
        nmsed_outs.append([c + 1, s, xmin, ymin, xmax, ymax])

    return nmsed_outs, num_det


def retinanet_detection_out(boxes_list, scores_list, anchors_list, im_info,
                            score_threshold, nms_threshold, nms_top_k,
                            keep_top_k):
    class_num = scores_list[0].shape[-1]
    im_height, im_width, im_scale = im_info

    num_level = len(scores_list)
    prediction = {}
    for lvl in range(num_level):
        scores_per_level = scores_list[lvl]
        scores_per_level = scores_per_level.flatten()
        bboxes_per_level = boxes_list[lvl]
        bboxes_per_level = bboxes_per_level.flatten()
        anchors_per_level = anchors_list[lvl]
        anchors_per_level = anchors_per_level.flatten()

        thresh = score_threshold if lvl < (num_level - 1) else 0.0
        selected_indices = np.argwhere(scores_per_level > thresh)
        scores = scores_per_level[selected_indices]
        sorted_indices = np.argsort(-scores, axis=0, kind='mergesort')
        if nms_top_k > -1 and nms_top_k < sorted_indices.shape[0]:
            sorted_indices = sorted_indices[:nms_top_k]

        for i in range(sorted_indices.shape[0]):
            idx = selected_indices[sorted_indices[i]]
            idx = idx[0][0]
            a = int(idx / class_num)
            c = int(idx % class_num)
            box_offset = a * 4
            anchor_box_width = anchors_per_level[
                box_offset + 2] - anchors_per_level[box_offset] + 1
            anchor_box_height = anchors_per_level[
                box_offset + 3] - anchors_per_level[box_offset + 1] + 1
            anchor_box_center_x = anchors_per_level[
                box_offset] + anchor_box_width / 2
            anchor_box_center_y = anchors_per_level[box_offset +
                                                    1] + anchor_box_height / 2

            target_box_center_x = bboxes_per_level[
                box_offset] * anchor_box_width + anchor_box_center_x
            target_box_center_y = bboxes_per_level[
                box_offset + 1] * anchor_box_height + anchor_box_center_y
            target_box_width = math.exp(bboxes_per_level[box_offset +
                                                         2]) * anchor_box_width
            target_box_height = math.exp(bboxes_per_level[
                box_offset + 3]) * anchor_box_height

            pred_box_xmin = target_box_center_x - target_box_width / 2
            pred_box_ymin = target_box_center_y - target_box_height / 2
            pred_box_xmax = target_box_center_x + target_box_width / 2 - 1
            pred_box_ymax = target_box_center_y + target_box_height / 2 - 1

            pred_box_xmin = pred_box_xmin / im_scale
            pred_box_ymin = pred_box_ymin / im_scale
            pred_box_xmax = pred_box_xmax / im_scale
            pred_box_ymax = pred_box_ymax / im_scale

            pred_box_xmin = max(
                min(pred_box_xmin, np.round(im_width / im_scale) - 1), 0.)
            pred_box_ymin = max(
                min(pred_box_ymin, np.round(im_height / im_scale) - 1), 0.)
            pred_box_xmax = max(
                min(pred_box_xmax, np.round(im_width / im_scale) - 1), 0.)
            pred_box_ymax = max(
                min(pred_box_ymax, np.round(im_height / im_scale) - 1), 0.)

            if c not in prediction.keys():
                prediction[c] = []
            prediction[c].append([
                pred_box_xmin, pred_box_ymin, pred_box_xmax, pred_box_ymax,
                scores_per_level[idx]
            ])

    nmsed_outs, nmsed_num = multiclass_nms(prediction, class_num, keep_top_k,
                                           nms_threshold)
    return nmsed_outs, nmsed_num


def batched_retinanet_detection_out(boxes, scores, anchors, im_info,
                                    score_threshold, nms_threshold, nms_top_k,
                                    keep_top_k):
    batch_size = scores[0].shape[0]
    det_outs = []
    lod = [0]

    for n in range(batch_size):
        boxes_per_batch = []
        scores_per_batch = []

        num_level = len(scores)
        for lvl in range(num_level):
            boxes_per_batch.append(boxes[lvl][n])
            scores_per_batch.append(scores[lvl][n])

        nmsed_outs, nmsed_num = retinanet_detection_out(
            boxes_per_batch, scores_per_batch, anchors, im_info[n],
            score_threshold, nms_threshold, nms_top_k, keep_top_k)
        lod.append(nmsed_num)
        if nmsed_num == 0:
            continue

        det_outs.extend(nmsed_outs)
    return det_outs, lod

