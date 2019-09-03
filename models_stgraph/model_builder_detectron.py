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
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.initializer import Normal
from paddle.fluid.initializer import MSRA
from paddle.fluid.regularizer import L2Decay
from config import cfg


class RetinaNet(object):
    def __init__(self,
                 add_conv_body_func=None,
                 add_fpn_neck_func=None,
                 anchor_strides=[8, 16, 32, 64, 128],
                 mode='train',
                 use_pyreader=True,
                 use_random=True):
        self.add_conv_body_func = add_conv_body_func
        self.add_fpn_neck_func = add_fpn_neck_func
        self.mode = mode
        self.use_pyreader = use_pyreader
        self.use_random = use_random
        self.anchor_strides = anchor_strides

    def build_model(self, image_shape):
        self.build_input(image_shape)
        # backbone
        feat_layers = self.extract_feat(self.image)
        # bbox head
        self.retina_heads(feat_layers)

        if self.mode != 'train':
            self.eval_bbox()
        

    def eval_bbox_out(self):
        return self.bbox_pred, self.pred_scores, self.im_info

    def build_input(self, image_shape):
        if self.use_pyreader:
            in_shapes = [[-1] + image_shape, [-1, 4], [-1, 1], [-1, 1],
                         [-1, 3], [-1, 1]]
            lod_levels = [0, 1, 1, 1, 0, 0]
            dtypes = [
                'float32', 'float32', 'int32', 'int32', 'float32', 'int32'
            ]
            self.py_reader = fluid.layers.py_reader(
                capacity=64,
                shapes=in_shapes,
                lod_levels=lod_levels,
                dtypes=dtypes,
                use_double_buffer=True)
            ins = fluid.layers.read_file(self.py_reader)
            self.image = ins[0]
            self.gt_box = ins[1]
            self.gt_label = ins[2]
            self.is_crowd = ins[3]
            self.im_info = ins[4]
            self.im_id = ins[5]
        else:
            self.image = fluid.layers.data(
                name='image', shape=image_shape, dtype='float32')
            self.gt_box = fluid.layers.data(
                name='gt_box', shape=[4], dtype='float32', lod_level=1)
            self.gt_label = fluid.layers.data(
                name='gt_label', shape=[1], dtype='int32', lod_level=1)
            self.is_crowd = fluid.layers.data(
                name='is_crowd', shape=[1], dtype='int32', lod_level=1)
            self.im_info = fluid.layers.data(
                name='im_info', shape=[3], dtype='float32')
            self.im_id = fluid.layers.data(
                name='im_id', shape=[1], dtype='int32')

    def feeds(self):
        if self.mode == 'infer':
            return [self.image, self.im_info]
        if self.mode == 'val':
            return [self.image, self.im_info, self.im_id]
        return [
            self.image, self.gt_box, self.gt_label, self.is_crowd,
            self.im_info, self.im_id
        ]

    def eval_bbox(self):
        scores = []
        
        for i in range(len(self.cls_score)):
            cls_prob = fluid.layers.sigmoid(self.cls_score[i])
            scores.append(cls_prob)
        
        self.pred_scores = scores

    def extract_feat(self, img):
        # resnet body
        print(self.image)
        self.body_conv = self.add_conv_body_func(self.image)
        # fpn neck
        if self.add_fpn_neck_func == None:
            return self.body_conv
        self.neck_conv = self.add_fpn_neck_func(self.body_conv, start_level=cfg.start_level)
        return self.neck_conv

    def cls_subnet(self, subnet_input, num_conv=4, lvl=3):
        subnet_blob = subnet_input
        for i in range(num_conv):
            suffix = 'n{}_fpn{}'.format(i, lvl)
            subnet_blob_in = subnet_blob
            subnet_blob = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=256,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                name='retnet_cls_conv_' + suffix,
                param_attr=ParamAttr('retnet_cls_conv_n{}_fpn{}_w'.format(i, 3), 
                                     initializer=Normal(loc=0., scale=0.01)),
                bias_attr=ParamAttr('retnet_cls_conv_n{}_fpn{}_b'.format(i, 3),
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.)))
            
        # cls prediction
        num_anchors = cfg.scales_per_octave * len(cfg.aspect_ratio)
        out_channel = (cfg.class_num - 1) * num_anchors
        # bias initialization: b = -log((1 - pai) / pai)
        prior_prob = 0.01 # pai
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        out = fluid.layers.conv2d(
            input=subnet_blob,
            num_filters=out_channel,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            name='retnet_cls_pred_fpn{}'.format(lvl),
            param_attr=ParamAttr(name='retnet_cls_pred_fpn{}_w'.format(3), 
                                 initializer=Normal(loc=0., scale=0.01)),           
            bias_attr=ParamAttr(name='retnet_cls_pred_fpn{}_b'.format(3),
                                initializer=Constant(value=bias_init),
                                learning_rate=2.,
                                regularizer=L2Decay(0.)))

        return out

    def box_subnet(self, subnet_input, num_conv=4, lvl=3):
        subnet_blob = subnet_input
        for i in range(num_conv):
            suffix = 'n{}_fpn{}'.format(i, lvl)
            subnet_blob_in = subnet_blob
            subnet_blob = fluid.layers.conv2d(
                input=subnet_blob_in,
                num_filters=256,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                name='retnet_bbox_conv_' + suffix,
                param_attr=ParamAttr(name='retnet_bbox_conv_n{}_fpn{}_w'.format(i, 3), 
                                     initializer=Normal(loc=0., scale=0.01)),
                bias_attr=ParamAttr(name='retnet_bbox_conv_n{}_fpn{}_b'.format(i, 3),
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.)))

        num_anchors = cfg.scales_per_octave * len(cfg.aspect_ratio)
        out_channel = 4 * num_anchors
        out = fluid.layers.conv2d(
            input=subnet_blob,
            num_filters=out_channel,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            name='retnet_bbox_pred_fpn{}'.format(lvl),
            param_attr=ParamAttr(name='retnet_bbox_pred_fpn{}_w'.format(3), 
                                 initializer=Normal(loc=0., scale=0.01)),           
            bias_attr=ParamAttr(name='retnet_bbox_pred_fpn{}_b'.format(3),
                                learning_rate=2.,
                                regularizer=L2Decay(0.)))

        return out


    def retina_heads(self, head_input):
        self.cls_score = []
        self.bbox_pred = []
        self.anchors = []
        self.vars = []
        self.bbox = []
        self.probs = []

        num_levels = len(head_input)
        for i in range(num_levels):
            self.cls_score.append(self.cls_subnet(head_input[i], 
                                                 cfg.num_convs, 
                                                 i+3))
            self.bbox_pred.append(self.box_subnet(head_input[i], 
                                                 cfg.num_convs, 
                                                 i+3))

            anchor_sizes = []
            for octave in range(cfg.scales_per_octave):
                anchor_sizes.append(self.anchor_strides[i] * (2 ** (float(octave) / float(cfg.scales_per_octave))) * cfg.anchor_scale)
            anchor = []
            var = []
            #print("input: {}".format(i))
            #print(head_input[i])
            #fluid.layers.Print(head_input[i], summarize= 10, message='The content of head_input: ')
            anchor, var = fluid.layers.anchor_generator(
                input=head_input[i],
                anchor_sizes=anchor_sizes,
                aspect_ratios=cfg.aspect_ratio,
                variance=cfg.variances,
                stride=[self.anchor_strides[i], self.anchor_strides[i]])
            #if i == (num_levels-1):
            #   fluid.layers.Print(anchor, message='The content of anchor: ')
            self.anchors.append(anchor)
            self.vars.append(var) 
            
    def get_convs(self):
        shapes = []
        rkeys = []
        if self.add_fpn_neck_func == None:
            for i in range(len(self.body_conv)):
                shapes.append(fluid.layer.shape(self.body_conv[i]))
                rkeys.append("shape_{}".format(str(i)))
        return shapes, rkeys

    def loss(self):
        cls_score = []
        bbox_pred = []
        anchor = []
        var = []
        bbox = []
        probs = []
        for i in range(len(self.cls_score)):
            cls_score_reshape = fluid.layers.transpose(
                self.cls_score[i], perm=[0, 2, 3, 1])
            bbox_pred_reshape = fluid.layers.transpose(
                self.bbox_pred[i], perm=[0, 2, 3, 1])
            #print(cls_score_reshape)
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
        '''
        cls_score_list = cls_score[-1]
        bbox_pred_list = bbox_pred[-1]
        anchor_list = anchor[-1]
        var_list = var[-1]
        '''
        #fluid.layers.Print(anchor_list, summarize=10, message='The content of anchor_list')
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
                straddle_thresh=cfg.TRAIN.straddle_thresh,
                positive_overlap=cfg.TRAIN.positive_overlap,
                negative_overlap=cfg.TRAIN.negative_overlap)
        
        fg_num = fluid.layers.reduce_sum(fg_num, name='fg_num')
        #fluid.layers.Print(fg_num, message='The content of fg_num')
        #fluid.layers.Print(score_tgt, summarize=50, message='The content of score_tgt')
        #fluid.layers.Print(loc_tgt, message='The content of loc_tgt')
        cls_loss = fluid.layers.sigmoid_focal_loss(
            x=score_pred, 
            label=score_tgt, 
            fg_num=fg_num,
            gamma=cfg.TRAIN.gamma,
            alpha=cfg.TRAIN.alpha,
            scale=(1/float(cfg.num_gpus)),
            num_classes=cfg.class_num - 1)
        loss_cls = fluid.layers.reduce_sum(
            cls_loss, name='loss_cls')
        #fluid.layers.Print(loss_cls,  message='The content of loss_cls: ')
        loss_cls.persistable = True       
        #fluid.layers.Print(loc_tgt, message='The content of loc_tgt: ')
        #fluid.layers.Print(loc_pred,  message='The content of loc_pred: ')
        reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0151134457776365,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        #fluid.layers.Print(reg_loss, summarize=10, message='The content of reg_loss: ')
        reg_loss = fluid.layers.scale(reg_loss, scale=(1/float(cfg.num_gpus)))
        reg_loss = fluid.layers.reduce_sum(
            reg_loss, name='loss_bbox')
        loss_bbox = reg_loss / fg_num
        loss_bbox.persistable = True
        #fluid.layers.Print(loss_bbox,  message='The content of loss_bbox: ')
        losses = [loss_cls, loss_bbox]
        rkeys = ['loss', 'loss_cls', 'loss_bbox']
        loss = fluid.layers.sum(losses)
        rloss = [loss] + losses
        return rloss, rkeys
