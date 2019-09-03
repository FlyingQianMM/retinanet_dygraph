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
        return self.pred_result

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
        anchors = []
        varis = []
        boxes = []
        scores = []
        self.im_scale = fluid.layers.slice(
            self.im_info, [1], starts=[2], ends=[3])
        
        for i in range(len(self.cls_score)):
            #fluid.layers.Print(self.cls_score[i], summarize=10, message="The content of cls_score[{}]: ".format(i))
            cls_prob = fluid.layers.sigmoid(self.cls_score[i])
            #fluid.layers.Print(cls_prob, summarize=10, message="The content of cls_prob[{}]: ".format(i))
            anchor = fluid.layers.reshape(self.anchors[i], shape=[-1,4])
            var = fluid.layers.reshape(self.vars[i], shape=[-1,4])
            bbox_pred_reshape = fluid.layers.transpose(
                self.bbox_pred[i], perm=[0, 2, 3, 1]) 
            bbox_pred_reshape = fluid.layers.reshape(bbox_pred_reshape, shape=[0,-1,4])
            '''
            decoded_box = fluid.layers.box_coder(
                prior_box=anchor,
                prior_box_var=cfg.variances,
                target_box=bbox_pred_reshape,
                code_type='decode_center_size',
                box_normalized=False,
                axis=1)
            decoded_box = decoded_box / self.im_scale
            decoded_box = fluid.layers.slice(decoded_box, axes=[0], starts=[0], ends = [80])
            fluid.layers.Print(decoded_box, summarize=10, message="The content of decoded_box: ")
            fluid.layers.Print(self.im_info, summarize=10, message="The content of im_info: ")
            #cliped_box = fluid.layers.box_clip(input=decoded_box, im_info=self.im_info)
            expand_box = fluid.layers.reshape(x=decoded_box, shape=(-1,1,4))
            expand_box = fluid.layers.expand(x=expand_box, expand_times=[1, cfg.class_num-1, 1])
            '''
            cls_score_reshape = fluid.layers.transpose(
                self.cls_score[i], perm=[0, 2, 3, 1]) 
            cls_score_reshape = fluid.layers.reshape(
                x=cls_score_reshape, shape=(0, -1, cfg.class_num - 1))

            boxes.append(bbox_pred_reshape)
            scores.append(cls_score_reshape)
            anchors.append(anchor)
            varis.append(var)
            
        anchor_list = fluid.layers.concat(anchors, axis=0)
        var_list = fluid.layers.concat(varis, axis=0) 
        boxes_list = fluid.layers.concat(boxes, axis=1)
        scores_list = fluid.layers.concat(scores, axis=1)
        #fluid.layers.Print(boxes_list, summarize=10, message="The content of boxes_list: ")
        #fluid.layers.Print(scores_list, summarize=10, message="The content of scores_list: ")
        '''
        self.pred_result = fluid.layers.multiclass_nms(
            bboxes=boxes_list,
            scores=scores_list,
            score_threshold=cfg.TEST.score_thresh,
            background_label=-1,
            nms_top_k=cfg.TEST.nms_top_k,
            nms_threshold=cfg.TEST.nms_thresh,
            keep_top_k=cfg.TEST.detections_per_im,
            normalized=False)
        fluid.layers.Print(self.pred_result, summarize=10, message="The content of pred_result: ")
        '''
        nmsed_out = fluid.layers.retina_detection_output(
            boxes_list, scores_list, anchor_list, var_list, self.im_scale, background_label=-1, nms_threshold=cfg.TEST.nms_thresh,
            nms_top_k=cfg.TEST.nms_top_k, keep_top_k=cfg.TEST.detections_per_im, score_threshold=cfg.TEST.score_thresh)
        #print(nmsed_out)
        #fluid.layers.Print(nmsed_out, summarize=18, message="The content of nmsed_out: ")
        self.pred_result = nmsed_out

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
                param_attr=ParamAttr(name='retnet_cls_conv_n{}_fpn{}_w'.format(i, 3), 
                                     initializer=Normal(loc=0., scale=0.01)),
                bias_attr=ParamAttr(name='retnet_cls_conv_n{}_fpn{}_b'.format(i, 3),
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
                                                 "fpn{}".format(i)))

            anchor_sizes = []
            for octave in range(cfg.scales_per_octave):
                anchor_sizes.append(self.anchor_strides[i] * (2 ** (float(octave) / float(cfg.scales_per_octave))) * cfg.anchor_scale)
            anchor = []
            var = []
            anchor, var = fluid.layers.anchor_generator(
                input=head_input[i],
                anchor_sizes=anchor_sizes,
                aspect_ratios=cfg.aspect_ratio,
                variance=cfg.variances,
                stride=[self.anchor_strides[i], self.anchor_strides[i]])
            self.anchors.append(anchor)
            self.vars.append(var) 
            
    def SuffixNet(self, conv5):
        mask_out = fluid.layers.conv2d_transpose(
            input=conv5,
            num_filters=cfg.dim_reduced,
            filter_size=2,
            stride=2,
            act='relu',
            param_attr=ParamAttr(
                name='conv5_mask_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name='conv5_mask_b', learning_rate=2., regularizer=L2Decay(0.)))
        act_func = None
        if self.mode != 'train':
            act_func = 'sigmoid'
        mask_fcn_logits = fluid.layers.conv2d(
            input=mask_out,
            num_filters=cfg.class_num,
            filter_size=1,
            act=act_func,
            param_attr=ParamAttr(
                name='mask_fcn_logits_w', initializer=MSRA(uniform=False)),
            bias_attr=ParamAttr(
                name="mask_fcn_logits_b",
                learning_rate=2.,
                regularizer=L2Decay(0.)))

        if self.mode != 'train':
            mask_fcn_logits = fluid.layers.lod_reset(mask_fcn_logits,
                                                     self.pred_result)
        return mask_fcn_logits

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
        
        reg_loss = fluid.layers.smooth_l1(
            x=loc_pred,
            y=loc_tgt,
            sigma=3.0,
            inside_weight=bbox_weight,
            outside_weight=bbox_weight)
        fluid.layers.Print(loc_pred, summarize=10, message='The content of loc_pred: ')
        fluid.layers.Print(reg_loss, summarize=10, message='The content of reg_loss: ')
        reg_loss = fluid.layers.scale(reg_loss, scale=(1/float(cfg.num_gpus)))
        reg_loss = fluid.layers.reduce_sum(
            reg_loss, name='loss_bbox')
        loss_bbox = reg_loss / fg_num
        
        losses = [loss_cls, loss_bbox]
        rkeys = ['loss', 'loss_cls', 'loss_bbox']
        loss = fluid.layers.sum(losses)
        rloss = [loss] + losses
        return rloss, rkeys
