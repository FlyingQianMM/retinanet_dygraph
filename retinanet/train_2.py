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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import numpy as np
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time
import collections

import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import reader
from models.model_builder import *
from config import cfg

def optimizer_setting(learning_rate, boundaries, values):
    start_lr = cfg.learning_rate * cfg.warm_up_factor
    
    lr = fluid.layers.exponential_with_warmup_decay(
        boundaries=boundaries,
        values=values,
        warmup_iter=cfg.warm_up_iter,
        start_lr=start_lr,
        end_lr=cfg.learning_rate)
    
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        #learning_rate=fluid.layers.piecewise_decay(
        #    boundaries=boundaries, values=values),
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    return optimizer

def eval_coco(mode, data):
    model.eval()
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

    for batch_id, batch_data in enumerate(data()):
        start = time.time()
        im_info = []
        for x in batch_data:
            tmp = []


def train():
    learning_rate = cfg.learning_rate

    if cfg.enable_ce:
        fluid.default_startup_program().random_seed = 1000
        fluid.default_main_program().random_seed = 1000
        import random
        random.seed(0)
        np.random.seed(0)

    devices = os.getenv("CUDA_VISIBLE_DEVICES") or ""
    devices_num = len(devices.split(","))
    total_batch_size = devices_num * cfg.TRAIN.im_per_batch

    use_random = True
    if cfg.enable_ce:
        use_random = False

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    if cfg.use_data_parallel:
        strategy = fluid.dygraph.parallel.prepare_context()

    retinanet = RetinaNet("retinanet")
    optimizer = optimizer_setting(learning_rate, boundaries, values)

    if cfg.use_data_parallel:
        retinanet = fluid.dygraph.parallel.DataParallel(retinanet, strategy)
   
    if cfg.pretrained_model:
        
        def if_exist(var):
            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
        pretrained, _ = fluid.dygraph.load_persistables(cfg.pretrained_model)
        retinanet.load_dict(pretrained)

    shuffle = True
    if cfg.enable_ce:
        shuffle = False
    
    if cfg.use_data_parallel:
        train_reader = reader.train(
            batch_size=cfg.TRAIN.im_per_batch,
            total_batch_size=total_batch_size,
            #batch_size=total_batch_size,
            shuffle=shuffle)
        train_reader = fluid.contrib.reader.distributed_batch_reader(
            train_reader)
    else:
        train_reader = reader.train(
            batch_size=total_batch_size,
            shuffle=shuffle)
    
    test_reader = reader.test(total_batch_size)

    def save_model(model_state, postfix, optimizer=None):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        #if os.path.isdir(model_path):
        #    shutil.rmtree(model_path)
        fluid.dygraph.save_persistables(model_state, model_path, optimizer)


    def train_loop():
        keys = ['loss', 'loss_cls', 'loss_bbox']
        train_stats = TrainingStats(cfg.log_window, keys)

        retinanet.train()
        for iter_id, data in enumerate(train_reader()):
            start_time = time.time()

            gt_max_num = 0
            batch_size = len(data)
            x = data[0]
            for x in data:
                #print(x[1].shape[0])
                if x[1].shape[0] > gt_max_num:
                    gt_max_num = x[1].shape[0]
            image_data = np.array(
                [x[0] for x in data]).astype('float32')
            #print('image: {} {}'.format(abs(image_data).sum(), image_data.shape))
            '''
            gt_box_data = np.array(
                [x[1] for x in data]).astype('float32')
            '''
            gt_box_data = np.zeros([batch_size, gt_max_num, 4])
            gt_label_data = np.zeros([batch_size, gt_max_num])
            is_crowd_data = np.ones([batch_size, gt_max_num])
            for batch_id, x in enumerate(data):
                gt_num = x[1].shape[0]
                gt_box_data[batch_id, 0:gt_num, :] = x[1]
                gt_label_data[batch_id, 0:gt_num] = x[2]
                is_crowd_data[batch_id, 0:gt_num] = x[3]
            gt_box_data = gt_box_data.astype('float32')
            gt_label_data = gt_label_data.astype('int32')
            is_crowd_data = is_crowd_data.astype('int32')
            '''
            gt_label_data = np.array(
                [x[2] for x in data]).astype('int32')
            is_crowd_data = np.array(
                [x[3] for x in data]).astype('int32')
            '''
            im_info_data = np.array(
                [x[4] for x in data]).astype('float32')
            im_id_data = np.array(
                [x[5] for x in data]).astype('int32')
            outputs= retinanet('train', image_data, im_info_data, \
                gt_box_data, gt_label_data, is_crowd_data)
            loss_cls = outputs['loss_cls']
            loss_bbox = outputs['loss_bbox']
            loss = outputs['loss']
            score_pred = outputs['score_pred']
            loc_pred = outputs['loc_pred']
            cls_pred_list = outputs['cls_score_list']
            bbox_pred_list = outputs['bbox_pred_list']
            cls_score = outputs['cls_score']
            bbox_pred = outputs['bbox_pred']
            loss_cls_data = loss_cls.numpy()
            loss_bbox_data = loss_bbox.numpy()
            loss_data = loss.numpy()

            if cfg.use_data_parallel:
                loss = retinanet.scale_loss(loss)
                loss.backward()
                retinanet.apply_collective_grads()
            else:
                #print('begin backward')
                loss.backward()
                #print('end backward')
            '''
            print('score_pred grad: {} {}'.format(abs(score_pred.gradient()).sum(), score_pred.gradient().shape))
            print('loc_pred grad: {} {}'.format(abs(loc_pred.gradient()).sum(), loc_pred.gradient().shape))
            for var in cls_pred_list:
                print('cls grad: {} {}'.format(abs(var.gradient()).sum(), var.gradient().shape))
            for var in bbox_pred_list:
                print('bbox grad: {} {}'.format(abs(var.gradient()).sum(), var.gradient().shape))
            for var in cls_score:
                print('cls grad: {} {}'.format(abs(var.gradient()).sum(), var.gradient().shape))
            for var in bbox_pred:
                print('bbox grad: {} {}'.format(abs(var.gradient()).sum(), var.gradient().shape))
            dy_grad_value = {}
            for param in retinanet.parameters():
                if param.name == 'retnet_cls_conv_n3_fpn3/Conv2D_0.retnet_cls_conv_n3_fpn3_w' or \
                   param.name == 'retnet_cls_conv_n2_fpn3/Conv2D_0.retnet_cls_conv_n2_fpn3_w' or \
                   param.name == 'retnet_cls_conv_n1_fpn3/Conv2D_0.retnet_cls_conv_n1_fpn3_w' or \
                   param.name == 'retnet_cls_conv_n0_fpn3/Conv2D_0.retnet_cls_conv_n0_fpn3_w' or \
                   param.name == 'retnet_cls_pred_fpn3/Conv2D_0.retnet_cls_pred_fpn3_w':
                    np_array = np.array(param._ivar._grad_ivar().value()
                                        .get_tensor())
                    dy_grad_value[param.name + core.grad_var_suffix(
                    )] = [abs(np_array).sum(), np_array.shape]
            for key, value in dy_grad_value.items():
                print('{key}: {value}'.format(key = key, value = value))
            '''
            '''           
            dy_grad_value = {}
            for param in retinanet.parameters():
                #if #param.name == 'retnet_cls_conv_n3_fpn3/Conv2D_0.retnet_cls_conv_n3_fpn3_w' or \
                   #param.name == 'retnet_cls_conv_n2_fpn3/Conv2D_0.retnet_cls_conv_n2_fpn3_w' or \
                   #param.name == 'retnet_cls_conv_n1_fpn3/Conv2D_0.retnet_cls_conv_n1_fpn3_w' or \
                   #param.name == 'retnet_cls_conv_n0_fpn3/Conv2D_0.retnet_cls_conv_n0_fpn3_w' or \
                 if param.name == 'retnet_cls_pred_fpn3/Conv2D_0.retnet_cls_pred_fpn3_w':
                    np_array = np.array(param._ivar._grad_ivar().value()
                                        .get_tensor())
                    dy_grad_value[param.name + core.grad_var_suffix(
                    )] = [abs(np_array).sum(), np_array.shape]
                    np_array = np.array(param._ivar.value().get_tensor())
                    dy_grad_value[param.name] = [abs(np_array).sum(), np_array.shape]
            for key, value in dy_grad_value.items():
                print('{key}: {value}'.format(key = key, value = value))
            '''
            optimizer.minimize(loss)
            retinanet.clear_gradients()

            outs = [loss_data, loss_cls_data, loss_bbox_data]
            stats = {k: v.mean() for k, v in zip(keys, outs)}
            train_stats.update(stats)
            logs = train_stats.log()
            lr = optimizer._global_learning_rate().numpy()
            end_time = time.time()
            strs = '{}, iter: {}, lr: {} {}, time: {:.3f}'.format(
                now_time(), iter_id, lr,
                logs, end_time - start_time)
            print(strs)
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model(retinanet.state_dict(), "model_iter{}".format(iter_id))
                #fluid.dygraph.save_persistables(retinanet.state_dict(), 'model_final', optimizer)
                #retinanet.eval()
                #eval_coco(retinanet, test_reader)
                #retinanet.train()
            if (iter_id + 1) == cfg.max_iter:
                break
    train_loop()
    #fluid.dygraph.save_persistables(retinanet.state_dict(), 'model_final')
    save_model(retinanet.state_dict(), 'model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id) \
        if cfg.use_data_parallel else fluid.CUDAPlace(0) \
        if cfg.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        if cfg.enable_ce:
            fluid.default_startup_program().random_seed = 1000
            fluid.default_main_program().random_seed = 1000
            import random
            random.seed(0)
            np.random.seed(0)
        train()
