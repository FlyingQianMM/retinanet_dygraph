from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import logging
from collections import defaultdict

from config import cfg
from generate_anchors import generate_anchors
import boxes as box_utils

logger = logging.getLogger(__name__)


def _create_cell_anchors():
    """
    Generate all types of anchors for all fpn levels/scales/aspect ratios.
    This function is called only once at the beginning of inference.
    """
    k_max, k_min = 7, 3
    scales_per_octave = cfg.scales_per_octave
    aspect_ratios = cfg.aspect_ratio
    anchor_scale = cfg.anchor_scale
    A = scales_per_octave * len(aspect_ratios)
    anchors = {}
    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = np.zeros((A, 4))
        a = 0
        for octave in range(scales_per_octave):
            octave_scale = 2 ** (octave / float(scales_per_octave))
            for aspect in aspect_ratios:
                anchor_sizes = (stride * octave_scale * anchor_scale, )
                anchor_aspect_ratios = (aspect, )
                cell_anchors[a, :] = generate_anchors(
                    stride=stride, sizes=anchor_sizes,
                    aspect_ratios=anchor_aspect_ratios)
                a += 1
        anchors[lvl] = cell_anchors
    return anchors


def detect_bbox(box_preds, cls_probs, im_info_v, data, num_id_to_cat_id_map):
    im_info = np.array(im_info_v)
    anchors = _create_cell_anchors()
    dts_res = []
    # here the boxes_all are [x0, y0, x1, y1, score]
    boxes_all = defaultdict(list)
    k_max, k_min = 7, 3
    A = cfg.scales_per_octave * len(cfg.aspect_ratio)
    cnt = 0

    image_id = int(data[0][-1])

    for lvl in range(k_min, k_max + 1):
        # create cell anchors array
        stride = 2. ** lvl
        cell_anchors = anchors[lvl]

        # fetch per level probability
        cls_prob = np.array(cls_probs[cnt])
        box_pred = np.array(box_preds[cnt])
        cls_prob = cls_prob.reshape((
            cls_prob.shape[0], A, int(cls_prob.shape[1] / A),
            cls_prob.shape[2], cls_prob.shape[3]))
        box_pred = box_pred.reshape((
            box_pred.shape[0], A, 4, box_pred.shape[2], box_pred.shape[3]))
        cnt += 1

        cls_prob_ravel = cls_prob.ravel()
        # In some cases [especially for very small img sizes], it's possible that
        # candidate_ind is empty if we impose threshold 0.05 at all levels. This
        # will lead to errors since no detections are found for this image. Hence,
        # for lvl 7 which has small spatial resolution, we take the threshold 0.0
        th = cfg.TEST.score_thresh if lvl < k_max else 0.0
        candidate_inds = np.where(cls_prob_ravel > th)[0]
        if (len(candidate_inds) == 0):
            continue

        pre_nms_topn = min(cfg.TEST.nms_top_k, len(candidate_inds))
        inds = np.argpartition(
            cls_prob_ravel[candidate_inds], -pre_nms_topn)[-pre_nms_topn:]
        inds = candidate_inds[inds]

        inds_5d = np.array(np.unravel_index(inds, cls_prob.shape)).transpose()
        classes = inds_5d[:, 2]
        anchor_ids, y, x = inds_5d[:, 1], inds_5d[:, 3], inds_5d[:, 4]
        scores = cls_prob[:, anchor_ids, classes, y, x]

        boxes = np.column_stack((x, y, x, y)).astype(dtype=np.float32)
        boxes *= stride
        boxes += cell_anchors[anchor_ids, :]

        
        box_deltas = box_pred[0, anchor_ids, :, y, x]
        pred_boxes = (
            box_utils.bbox_transform(boxes, box_deltas)
            )
        im_scale = im_info[0][2]
        pred_boxes /= im_scale
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im_info[0][0:2])
        box_scores = np.zeros((pred_boxes.shape[0], 5))
        box_scores[:, 0:4] = pred_boxes
        box_scores[:, 4] = scores

        for cls in range(1, cfg.class_num):
            inds = np.where(classes == cls - 1)[0]
            if len(inds) > 0:
                boxes_all[cls].extend(box_scores[inds, :])

    # Combine predictions across all levels and retain the top scoring by class
    detections = []
    for cls, boxes in boxes_all.items():
        cls_dets = np.vstack(boxes).astype(dtype=np.float32)
        keep = box_utils.nms(cls_dets, cfg.TEST.nms_thresh)
        cls_dets = cls_dets[keep, :]
        out = np.zeros((len(keep), 6))
        out[:, 0:5] = cls_dets
        out[:, 5].fill(cls)
        detections.append(out)

    # detections (N, 6) format:
    #   detections[:, :4] - boxes
    #   detections[:, 4] - scores
    #   detections[:, 5] - classes
    detections = np.vstack(detections)
    # sort all again
    inds = np.argsort(-detections[:, 4])
    detections = detections[inds[0:cfg.TEST.detections_per_im], :]
     
    for i in range(detections.shape[0]):
        dt = detections[i]
        xmin, ymin, xmax, ymax, score, num_id = dt.tolist()
        category_id = num_id_to_cat_id_map[num_id]
        w = xmax - xmin + 1
        h = ymax - ymin + 1
        bbox = [xmin, ymin, w, h]
        dt_res = {
            'image_id': image_id,
            'category_id': category_id,
            'bbox': bbox,
            'score': score
        }
        dts_res.append(dt_res)
    return dts_res

