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
import sys
import numpy as np
import time
import shutil
from utility import parse_args, print_arguments, SmoothedValue, TrainingStats, now_time
import collections

import paddle
import paddle.fluid as fluid
import reader
import models.model_builder as model_builder
import models.resnet as resnet
import models.fpn as fpn
from learning_rate import exponential_with_warmup_decay
from config import cfg

var_name_list = ['read_file_0.tmp_0', 'res3d.add.output.5.tmp_0', 'res4f.add.output.5.tmp_0', 'res5c.add.output.5.tmp_0', 'fpn_inner_res3_3_sum_lateral.tmp_1', 'fpn_inner_res4_5_sum_lateral.tmp_1', 'fpn_inner_res5_2_sum.tmp_1', 'fpn_inner_res4_5_sum_topdown.tmp_0', 'fpn_inner_res4_5_sum', 'fpn_inner_res3_3_sum_topdown.tmp_0', 'fpn_inner_res3_3_sum', 'fpn_res3_3_sum.tmp_1', 'fpn_res4_5_sum.tmp_1', 'fpn_res5_2_sum.tmp_1', 'fpn_6.tmp_1', 'fpn_7.tmp_1', 'retnet_cls_conv_n0_fpn3.tmp_2', 'retnet_cls_conv_n1_fpn3.tmp_2', 'retnet_cls_conv_n2_fpn3.tmp_2', 'retnet_cls_conv_n3_fpn3.tmp_2', 'retnet_cls_pred_fpn3.tmp_1', 'retnet_bbox_conv_n0_fpn3.tmp_2', 'retnet_bbox_conv_n1_fpn3.tmp_2', 'retnet_bbox_conv_n2_fpn3.tmp_2', 'retnet_bbox_conv_n3_fpn3.tmp_2', 'retnet_bbox_pred_fpn3.tmp_1', 'retnet_cls_conv_n0_fpn4.tmp_2', 'retnet_cls_conv_n1_fpn4.tmp_2', 'retnet_cls_conv_n2_fpn4.tmp_2', 'retnet_cls_conv_n3_fpn4.tmp_2', 'retnet_cls_pred_fpn4.tmp_1', 'retnet_bbox_conv_n0_fpn4.tmp_2', 'retnet_bbox_conv_n1_fpn4.tmp_2', 'retnet_bbox_conv_n2_fpn4.tmp_2', 'retnet_bbox_conv_n3_fpn4.tmp_2', 'retnet_bbox_pred_fpn4.tmp_1', 'retnet_cls_conv_n0_fpn5.tmp_2', 'retnet_cls_conv_n1_fpn5.tmp_2', 'retnet_cls_conv_n2_fpn5.tmp_2', 'retnet_cls_conv_n3_fpn5.tmp_2', 'retnet_cls_pred_fpn5.tmp_1', 'retnet_bbox_conv_n0_fpn5.tmp_2', 'retnet_bbox_conv_n1_fpn5.tmp_2', 'retnet_bbox_conv_n2_fpn5.tmp_2', 'retnet_bbox_conv_n3_fpn5.tmp_2', 'retnet_bbox_pred_fpn5.tmp_1', 'retnet_cls_conv_n0_fpn6.tmp_2', 'retnet_cls_conv_n1_fpn6.tmp_2', 'retnet_cls_conv_n2_fpn6.tmp_2', 'retnet_cls_conv_n3_fpn6.tmp_2', 'retnet_cls_pred_fpn6.tmp_1', 'retnet_bbox_conv_n0_fpn6.tmp_2', 'retnet_bbox_conv_n1_fpn6.tmp_2', 'retnet_bbox_conv_n2_fpn6.tmp_2', 'retnet_bbox_conv_n3_fpn6.tmp_2', 'retnet_bbox_pred_fpn6.tmp_1', 'retnet_cls_conv_n0_fpn7.tmp_2', 'retnet_cls_conv_n1_fpn7.tmp_2', 'retnet_cls_conv_n2_fpn7.tmp_2', 'retnet_cls_conv_n3_fpn7.tmp_2', 'retnet_cls_pred_fpn7.tmp_1', 'retnet_bbox_conv_n0_fpn7.tmp_2', 'retnet_bbox_conv_n1_fpn7.tmp_2', 'retnet_bbox_conv_n2_fpn7.tmp_2', 'retnet_bbox_conv_n3_fpn7.tmp_2', 'retnet_bbox_pred_fpn7.tmp_1', 'retinanet_target_assign_0.tmp_4', 'retinanet_target_assign_0.tmp_3']

#, 'retinanet_target_assign_0.tmp_0', 'retinanet_target_assign_0.tmp_1', 'retinanet_target_assign_0.tmp_2', 'retinanet_target_assign_0.tmp_3', 'retinanet_target_assign_0.tmp_4', 'retinanet_target_assign_0.tmp_5', 'gather_0.tmp_0', 'gather_1.tmp_0', 'fg_num.tmp_0']

def train():
    learning_rate = cfg.learning_rate
    image_shape = [3, cfg.TRAIN.max_size, cfg.TRAIN.max_size]

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
    model = model_builder.RetinaNet(
        add_conv_body_func=resnet.add_ResNet50_conv_body,
        add_fpn_neck_func=fpn.add_fpn_neck,
        anchor_strides=[8, 16, 32, 64, 128],
        use_pyreader=cfg.use_pyreader,
        use_random=use_random)
    model.build_model(image_shape)
    losses, keys = model.loss()
    loss = losses[0]
    fetch_list = losses 

    boundaries = cfg.lr_steps
    gamma = cfg.lr_gamma
    step_num = len(cfg.lr_steps)
    values = [learning_rate * (gamma**i) for i in range(step_num + 1)]

    lr = exponential_with_warmup_decay(
        learning_rate=learning_rate,
        boundaries=boundaries,
        values=values,
        warmup_iter=cfg.warm_up_iter,
        warmup_factor=cfg.warm_up_factor)
    optimizer = fluid.optimizer.Momentum(
        learning_rate=lr,
        regularization=fluid.regularizer.L2Decay(cfg.weight_decay),
        momentum=cfg.momentum)
    optimizer.minimize(loss)
    fetch_list = fetch_list + [lr]

    #fluid.memory_optimize(
    #    fluid.default_main_program(), skip_opt_set=set(fetch_list))

    place = fluid.CUDAPlace(0) if cfg.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    #print(fluid.default_main_program().to_string(True))
    
    if cfg.pretrained_model:
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
        params = np.load(os.path.join('detectron-model', 'model_iter1.pkl'))
        params = params['blobs']

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
            #else:
                #print(var_name)
            #    if 'inner' in var:
            #        var_name = var[:14] + chr(ord("a")+int(var[15])) + var[16:]
            #    else:
            #        var_name = var[:8] + chr(ord("a")+int(var[9])) + var[10:]
            # paddle path
            #print(var_name)
            if os.path.exists('output_duiqi/init_model/'+var_name) and var_name not in temp:
                cnt += 1
                #print(var_name)
                temp.add(var_name)
                param = fluid.global_scope().find_var(var_name).get_tensor()
                if var == 'bbox_pred_w' or var=='cls_score_w' or 'fc' in var:
                    params[var] = np.transpose(params[var])
                param.set(params[var],place)
            #else:
                #print('not exist: ', var_name)
        print('load weight sum: ',cnt) 
#    if cfg.pretrained_model:
#
#        def if_exist(var):
#            return os.path.exists(os.path.join(cfg.pretrained_model, var.name))
#
#        fluid.io.load_vars(exe, cfg.pretrained_model, predicate=if_exist)
#
    

    if cfg.parallel:
       	build_strategy = fluid.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_threads = 1
        build_strategy.enable_inplace = False
        build_strategy.memory_optimize = False
        train_exe = fluid.ParallelExecutor(
            use_cuda=bool(cfg.use_gpu), 
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy)

    shuffle = True
    if cfg.enable_ce:
        shuffle = False
    if cfg.use_pyreader:
        train_reader = reader.train(
            batch_size=cfg.TRAIN.im_per_batch,
            total_batch_size=total_batch_size,
            padding_total=cfg.TRAIN.padding_minibatch,
            shuffle=shuffle)
        py_reader = model.py_reader
        py_reader.decorate_paddle_reader(train_reader)
    else:
        train_reader = reader.train(
            batch_size=total_batch_size, shuffle=shuffle)
        feeder = fluid.DataFeeder(place=place, feed_list=model.feeds())

    def save_model(postfix):
        model_path = os.path.join(cfg.model_save_dir, postfix)
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)
        fluid.io.save_persistables(exe, model_path)

	#print(fluid.default_main_program().to_string(True))
    for var in fluid.default_main_program().list_vars():
        #print(var.name)
        for debug_name in var_name_list:
            if var.name == debug_name:
                var.persistable = True
                fetch_list = fetch_list + [var]	

    def train_loop_pyreader():
        py_reader.start()
        train_stats = TrainingStats(cfg.log_window, keys)
        try:
            start_time = time.time()
            prev_start_time = start_time
            for iter_id in range(cfg.max_iter):
                if iter_id < 2:
                    continue
                prev_start_time = start_time
                start_time = time.time()
                outs = train_exe.run(fetch_list=[v.name for v in fetch_list],return_numpy=False)
                for i in range(len(var_name_list)):
                    if iter_id == (cfg.max_iter-1):
                        print(var_name_list[i], np.array(outs[i-len(var_name_list)]).mean(), np.array(outs[i-len(var_name_list)]).shape)
                    '''
                    if iter_id == (cfg.max_iter-1) and var_name_list[i] == 'retnet_bbox_pred_fpn7.tmp_1':
                        np.savetxt("{}_retnet_bbox_pred_fpn7_tmp_1.txt".format(iter_id), np.array(outs[i-len(var_name_list)]).flatten())
                    if iter_id == (cfg.max_iter-1) and var_name_list[i] == 'retinanet_target_assign_0.tmp_4':
                        np.savetxt("{}_retinanet_target_assign_0_tmp_4.txt".format(iter_id), np.array(outs[i-len(var_name_list)]).flatten())
                    if iter_id == (cfg.max_iter-1) and var_name_list[i] == 'retinanet_target_assign_0.tmp_3':
                        np.savetxt("{}_retinanet_target_assign_0_tmp_3.txt".format(iter_id), np.array(outs[i-len(var_name_list)]).flatten()) 
                    '''
                stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
                train_stats.update(stats)
                logs = train_stats.log()
                strs = '{}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                    now_time(), iter_id,
                    np.mean(outs[-1-len(var_name_list)]), logs, start_time - prev_start_time)
                print(strs)
                sys.stdout.flush()
                if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                    save_model("model_iter{}".format(iter_id))
            end_time = time.time()
            total_time = end_time - start_time
            last_loss = np.array(outs[0]).mean()
            if cfg.enable_ce:
                gpu_num = devices_num
                epoch_idx = iter_id + 1
                loss = last_loss
                print("kpis\teach_pass_duration_card%s\t%s" %
                      (gpu_num, total_time / epoch_idx))
                print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, loss))
        except (StopIteration, fluid.core.EOFException):
            py_reader.reset()

    def train_loop():
        start_time = time.time()
        prev_start_time = start_time
        start = start_time
        train_stats = TrainingStats(cfg.log_window, keys)
        for iter_id, data in enumerate(train_reader()):
            prev_start_time = start_time
            start_time = time.time()
            outs = train_exe.run(fetch_list=[v.name for v in fetch_list],
                                 feed=feeder.feed(data))
            stats = {k: np.array(v).mean() for k, v in zip(keys, outs[:-1])}
            train_stats.update(stats)
            logs = train_stats.log()
            strs = '{}, iter: {}, lr: {:.5f}, {}, time: {:.3f}'.format(
                now_time(), iter_id,
                np.mean(outs[-1]), logs, start_time - prev_start_time)
            print(strs)
            #print(outs[3:6])
            sys.stdout.flush()
            if (iter_id + 1) % cfg.TRAIN.snapshot_iter == 0:
                save_model("model_iter{}".format(iter_id))
            if (iter_id + 1) == cfg.max_iter:
                break
        end_time = time.time()
        total_time = end_time - start_time
        last_loss = np.array(outs[0]).mean()
        # only for ce
        if cfg.enable_ce:
            gpu_num = devices_num
            epoch_idx = iter_id + 1
            loss = last_loss
            print("kpis\teach_pass_duration_card%s\t%s" %
                  (gpu_num, total_time / epoch_idx))
            print("kpis\ttrain_loss_card%s\t%s" % (gpu_num, loss))

        return np.mean(every_pass_loss)

    if cfg.use_pyreader:
        train_loop_pyreader()
    else:
        train_loop()
    save_model('model_final')


if __name__ == '__main__':
    args = parse_args()
    print_arguments(args)
    train()
