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
import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg
from resnet import *
from fpn import *
from ops import *
from detection_output import *
import time


class cls_subnet(fluid.dygraph.Layer):
    def __init__(self,
                 namescope,
                 num_conv=4,
                 lvl=3):
        super(cls_subnet, self).__init__(namescope)

        self.subnet_conv_list = []
        self.num_conv = num_conv
        for i in range(num_conv):
            suffix = 'n{}_fpn{}'.format(i, lvl)
            subnet_conv = self.add_sublayer(
                'retnet_cls_conv_' + suffix,
                Conv2D('retnet_cls_conv_' + suffix,
                       num_filters=256,
                       filter_size=3,
                       stride=1,
                       padding=1,
                       act='relu',
                       param_attr=ParamAttr('retnet_cls_conv_n{}_fpn{}_w'.format(i, 3),
                                            initializer=Normal(loc=0., scale=0.01)),
                       bias_attr=ParamAttr('retnet_cls_conv_n{}_fpn{}_b'.format(i, 3),
                                           learning_rate=2.,
                                           regularizer=L2Decay(0.))))
            self.subnet_conv_list.append(subnet_conv)
        
        num_anchors = cfg.scales_per_octave * len(cfg.aspect_ratio)
        out_channel = (cfg.class_num - 1) * num_anchors
        prior_prob = 0.01 # pai
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        self.cls_pred = Conv2D('retnet_cls_pred_fpn{}'.format(3),
            num_filters=out_channel,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(name='retnet_cls_pred_fpn{}_w'.format(3),
                                 initializer=Normal(loc=0., scale=0.01)),
            bias_attr=ParamAttr(name='retnet_cls_pred_fpn{}_b'.format(3),
                                initializer=Constant(value=bias_init),
                                learning_rate=2.,
                                regularizer=L2Decay(0.)))

    def forward(self, inputs):
        subnet_blob = self.subnet_conv_list[0](inputs)
        for i in range(1, self.num_conv):
           subnet_blob = self.subnet_conv_list[i](subnet_blob)
        print(abs(subnet_blob.numpy()).sum())
        out = self.cls_pred(subnet_blob)
        return out

class box_subnet(fluid.dygraph.Layer):
    def __init__(self,
                 namescope,
                 num_conv=4,
                 lvl=3):
        super(box_subnet, self).__init__(namescope)

        self.subnet_conv_list = []
        self.num_conv = num_conv
        for i in range(num_conv):
            suffix = 'n{}_fpn{}'.format(i, lvl)
            subnet_conv = self.add_sublayer(
                'retnet_bbox_conv_' + suffix,
                Conv2D('retnet_bbox_conv_' + suffix,
                       num_filters=256,
                       filter_size=3,
                       stride=1,
                       padding=1,
                       act='relu',
                       param_attr=ParamAttr(name='retnet_bbox_conv_n{}_fpn{}_w'.format(i, 3),
                                            initializer=Normal(loc=0., scale=0.01)),
                       bias_attr=ParamAttr(name='retnet_bbox_conv_n{}_fpn{}_b'.format(i, 3),
                                           learning_rate=2.,
                                           regularizer=L2Decay(0.))))
            self.subnet_conv_list.append(subnet_conv)                                                             

        num_anchors = cfg.scales_per_octave * len(cfg.aspect_ratio)
        out_channel = 4 * num_anchors
        self.box_pred = Conv2D('retnet_bbox_pred_fpn{}'.format(lvl),
            num_filters=out_channel,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(name='retnet_bbox_pred_fpn{}_w'.format(3),
                                 initializer=Normal(loc=0., scale=0.01)),
            bias_attr=ParamAttr(name='retnet_bbox_pred_fpn{}_b'.format(3),
                                learning_rate=2.,
                                regularizer=L2Decay(0.)))
        
    def forward(self, inputs):
        subnet_blob = self.subnet_conv_list[0](inputs)
        for i in range(1, self.num_conv):
           subnet_blob = self.subnet_conv_list[i](subnet_blob)
        out = self.box_pred(subnet_blob)
 
        return out

class RetinaNet(fluid.dygraph.Layer):
    def __init__(self,
                 namescope,
                 anchor_strides=[8, 16, 32, 64, 128]):
        super(RetinaNet, self).__init__(namescope)

        self.add_conv_body_func = add_ResNet50_conv_body("resnet")
        self.add_fpn_neck_func = add_fpn_neck("fpn")
        num_levels = len(anchor_strides)
        self.anchor_strides = anchor_strides
        '''       
        self.subnet_cls_list = []
        for i in range(num_levels):
            subnet_cls = self.add_sublayer(
                "subnet_cls_{}".format(i + 3),
                cls_subnet("subnet_cls_{}".format(i + 3),
                           cfg.num_convs,
                           i+3))
            self.subnet_cls_list.append(subnet_cls)
        
        self.subnet_box_list = []
        for i in range(num_levels):
            subnet_box = self.add_sublayer(
                "subnet_box_{}".format(i + 3),
                box_subnet("subnet_box_{}".format(i + 3),
                           cfg.num_convs,
                           i+3))
            self.subnet_box_list.append(subnet_box)
        '''
        self.subnet_cls = cls_subnet("subnet_cls_{}".format(3),
                cfg.num_convs, 3)

        self.subnet_box = box_subnet("subnet_box_{}".format(3),
                cfg.num_convs, 3)
 
    def get_loss(self):
        
        cls_score = []
        bbox_pred = []
        anchor = []
        var = []

        for i in range(len(self.cls_score)):
            cls_score_reshape = fluid.layers.transpose(
                self.cls_score[i], perm=[0, 2, 3, 1])
            bbox_pred_reshape = fluid.layers.transpose(
                self.bbox_pred[i], perm=[0, 2, 3, 1])
            anchor_reshape = fluid.layers.reshape(self.anchors[i], shape=(-1, 4))
            var_reshape = fluid.layers.reshape(self.vars[i], shape=(-1, 4))
            cls_score_reshape = fluid.layers.reshape(
                x=cls_score_reshape, shape=(0, -1, cfg.class_num - 1))
            bbox_pred_reshape = fluid.layers.reshape(
                x=bbox_pred_reshape, shape=(0, -1, 4))
            cls_score.append(cls_score_reshape)
            bbox_pred.append(bbox_pred_reshape)
            anchor.append(anchor_reshape)
            var.append(var_reshape)
        cls_score_list = fluid.layers.concat(cls_score, axis=1)
        bbox_pred_list = fluid.layers.concat(bbox_pred, axis=1)
        anchor_list = fluid.layers.concat(anchor, axis=0)
        var_list = fluid.layers.concat(var, axis=0)

	    #start = time.time()
        score_pred, loc_pred, score_tgt, loc_tgt, bbox_weight, fg_num = \
            fluid.layers.retinanet_target_assign(
            bbox_pred=bbox_pred_list,
            cls_logits=cls_score_list,
            anchor_box=anchor_list,
            anchor_var=var_list,
            gt_boxes=self.gt_box,
            gt_labels=self.gt_label,
            is_crowd=self.is_crowd,
            im_info=self.im_info,
            num_classes=cfg.class_num - 1,
            positive_overlap=cfg.TRAIN.positive_overlap,
            negative_overlap=cfg.TRAIN.negative_overlap)
        #end = time.time()
        #total_time = end - start
        #print('python op time: {}'.format(end-start))

        fg_num = fluid.layers.reduce_sum(fg_num, name='fg_num')
        fg_num._stop_gradient = True
        if cfg.enable_ce:
            print('fg_num: {} {}'.format(abs(fg_num.numpy()).sum(), fg_num.numpy().shape))
        cls_loss = fluid.layers.sigmoid_focal_loss(
            x=score_pred, 
            label=score_tgt, 
            fg_num=fg_num,
            gamma=cfg.TRAIN.gamma,
            alpha=cfg.TRAIN.alpha)
        if cfg.enable_ce:
            print('cls_loss: {} {}'.format(abs(cls_loss.numpy()).sum(), cls_loss.numpy().shape))
        loss_cls = fluid.layers.reduce_sum(
            cls_loss, name='loss_cls')
        if cfg.enable_ce:
            print('loss_cls: {} {}'.format(abs(loss_cls.numpy()).sum(), loss_cls.numpy().shape))
        #loss_cls.persistable = True       
        reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=cfg.TRAIN.sigma,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        if cfg.enable_ce:
            print('reg_loss: {} {}'.format(abs(reg_loss.numpy()).sum(), reg_loss.numpy().shape))
        reg_loss = fluid.layers.reduce_sum(
            reg_loss, name='loss_bbox')
        if cfg.enable_ce:
            print('reg_loss: {} {}'.format(abs(reg_loss.numpy()).sum(), reg_loss.numpy().shape))
        loss_bbox = reg_loss / fg_num
        if cfg.enable_ce:
            print('loss_bbox: {} {}'.format(abs(loss_bbox.numpy()).sum(), loss_bbox.numpy().shape))
        #loss_bbox.persistable = True
        losses = [loss_cls, loss_bbox]
        loss = fluid.layers.sum(losses)
        if cfg.enable_ce:
            print('loss in the forward propogation : {}'.format(abs(loss.numpy()).sum()))
        out = {}
        out['loss'] = loss
        out['loss_cls'] = loss_cls
        out['loss_bbox'] = loss_bbox
        out['score_pred'] = score_pred
        out['loc_pred'] = loc_pred
        out['cls_score_list'] = cls_score
        out['bbox_pred_list'] = bbox_pred
        out['cls_score'] = self.cls_score
        out['bbox_pred'] = self.bbox_pred
        return out


    def eval_prediction(self):
        anchors = []
        boxes = []
        scores = []
        
        for i in range(len(self.cls_score)):
            cls_prob = fluid.layers.sigmoid(self.cls_score[i])
            anchor = fluid.layers.reshape(self.anchors[i], shape=[-1,4])
            var = fluid.layers.reshape(self.vars[i], shape=[-1,4])
            bbox_pred_reshape = fluid.layers.transpose(
                self.bbox_pred[i], perm=[0, 2, 3, 1]) 
            bbox_pred_reshape = fluid.layers.reshape(bbox_pred_reshape, shape=[0,-1,4])
            cls_score_reshape = fluid.layers.transpose(
                cls_prob, perm=[0, 2, 3, 1]) 
            cls_score_reshape = fluid.layers.reshape(
                x=cls_score_reshape, shape=(0, -1, cfg.class_num - 1))

            boxes.append(bbox_pred_reshape.numpy())
            scores.append(cls_score_reshape.numpy())
            anchors.append(anchor.numpy())
        
        det_outs, lod = batched_retinanet_detection_out(
           boxes = boxes,
           scores = scores,
           anchors = anchors,
           im_info = self.im_info,
           score_threshold = cfg.TEST.score_thresh,
           nms_top_k = cfg.TEST.nms_top_k,
           keep_top_k = cfg.TEST.detections_per_im,
           nms_threshold = cfg.TEST.nms_thresh)
        prediction = {}
        prediction['det_outs'] = det_outs
        prediction['lod'] = lod
        return prediction


    def forward(self, mode, image, im_info, gt_box=None, gt_label=None, is_crowd=None):
        inputs = to_variable(image)
        body_layers = self.add_conv_body_func(inputs)
        neck_layers = self.add_fpn_neck_func(body_layers)
        self.cls_score = []
        self.bbox_pred = []
        self.anchors = []
        self.vars = []
        print("im_info: {}".format(im_info))
        self.im_info = to_variable(im_info)
        self.gt_box = to_variable(gt_box)
        self.gt_label = to_variable(gt_label)
        self.is_crowd = to_variable(is_crowd)

        num_levels = len(neck_layers)
        for i in range(num_levels):
            self.cls_score.append(
                #self.subnet_cls_list[i](neck_layers[i]))
                self.subnet_cls(neck_layers[i]))
            #print('cls pred {} : {} {}'.format(i+3, self.cls_score[i].numpy().sum(), self.cls_score[i].numpy().shape))
            ##print('cls pred grad {} : {} {}'.format(i+3, self.cls_score[i].gradient().sum(), self.cls_score[i].gradient().shape))
            self.bbox_pred.append(
                #self.subnet_box_list[i](neck_layers[i]))
                self.subnet_box(neck_layers[i]))
            #print('box pred {} : {} {}'.format(i+3, self.bbox_pred[i].numpy().sum(), self.bbox_pred[i].numpy().shape))
            ##print('box pred grad {} : {} {}'.format(i+3, self.bbox_pred[i].gradient().sum(), self.bbox_pred[i].gradient().shape))
            anchor_sizes = []
            for octave in range(cfg.scales_per_octave):
                anchor_sizes.append(self.anchor_strides[i] * (2 ** (float(octave) \
                    / float(cfg.scales_per_octave))) * cfg.anchor_scale)
            anchor = []
            var = []
            anchor, var = fluid.layers.anchor_generator(
                input=neck_layers[i],
                anchor_sizes=anchor_sizes,
                aspect_ratios=cfg.aspect_ratio,
                variance=cfg.variances,
                stride=[self.anchor_strides[i], self.anchor_strides[i]])
            self.anchors.append(anchor)
            self.vars.append(var)
        if mode == 'train':
            return self.get_loss()
        elif mode == 'eval':
            return self.eval_prediction()

