import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Normal
from config import cfg


def add_fpn_neck(inputs, 
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 ):
    inputs_name = ['res3_3_sum', 'res4_5_sum', 'res5_2_sum']
    if end_level == -1:
        end_level = len(inputs)

    # build laterals
    lateral_convs = []
    for i in range(start_level, end_level):
        if i < (end_level-1):
            lateral_name = 'fpn_inner_' + inputs_name[i] + '_lateral'
        else:
            lateral_name = 'fpn_inner_' + inputs_name[i]
        conv = fluid.layers.conv2d(
            input=inputs[i],
            num_filters=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name=lateral_name + '_w', 
                                 initializer=Xavier()),
            bias_attr=ParamAttr(name=lateral_name + '_b', 
                                 initializer=Constant(value=0.0),
                                 learning_rate=2.,
                                 regularizer=L2Decay(0.)),
            name=lateral_name)
        lateral_convs.append(conv)
    
    # build top-down path
    used_body_levels = len(lateral_convs)
    '''
    top_down_convs = []
    for i in range(used_body_levels - 1, 0, -1):
        conv = fluid.layers.resize_nearest(lateral_convs[i], 
                                          scale=2, 
                                          name="res{}_top-down".format(i+3+start_level))
        print(conv, lateral_convs[i - 1])
        top_down_convs.append(conv)
    '''
    #print("used_body_levels: {}".format(used_body_levels))
    for i in range(used_body_levels - 1, 0, -1):
        '''       
        lateral_shape = fluid.layers.shape(lateral_convs[i - 1])
        lateral_shape = fluid.layers.slice(lateral_shape, axes=[0], starts=[2], ends=[4])
        lateral_shape.stop_gradient = True
        fpn_inner_name = 'fpn_inner_' + inputs_name[i-1]
        topdown_name = fpn_inner_name + '_topdown'
        top_down = fluid.layers.resize_nearest(lateral_convs[i], 
                                               scale=2.,
                                               actual_shape=lateral_shape,
                                               name=topdown_name)
        '''
        shape = fluid.layers.shape(lateral_convs[i])
        shape_hw = fluid.layers.slice(shape, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * 2
        out_shape.stop_gradient = True
        fpn_inner_name = 'fpn_inner_' + inputs_name[i-1]
        topdown_name = fpn_inner_name + '_topdown'
        top_down = fluid.layers.resize_nearest(lateral_convs[i], 
                                               scale=2.,
                                               actual_shape=out_shape,
                                               name=topdown_name)
        
        lateral_convs[i - 1] = fluid.layers.elementwise_add(
            x=lateral_convs[i - 1], 
            y=top_down,
            name=fpn_inner_name
            )
        
    # build outputs
    # part 1: from original levels
    outs = []
    for i in range(used_body_levels):
        fpn_name = 'fpn_' + inputs_name[i]
        conv = fluid.layers.conv2d(
            input=lateral_convs[i],
            num_filters=out_channels,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(name=fpn_name+'_w', 
                                 initializer=Xavier()),
            bias_attr=ParamAttr(name=fpn_name+'_b', 
                                 initializer=Constant(value=0.0),
                                 learning_rate=2.,
                                 regularizer=L2Decay(0.)),
            name=fpn_name)
        outs.append(conv)

    # part 2: add extra levels
    if(num_outs > used_body_levels):
        fpn_blob = inputs[end_level - 1]
        fpn_blob_in = fpn_blob
        fpn_blob = fluid.layers.conv2d(
                 input=fpn_blob_in,
                 num_filters=out_channels,
                 filter_size=3,
                 stride=2,
                 padding=1,
                 param_attr=ParamAttr(name='fpn_{}_w'.format(6),
                                      initializer=Xavier()),
                 bias_attr=ParamAttr(name='fpn_{}_b'.format(6), 
                                      initializer=Constant(value=0.0),
                                      learning_rate=2.,
                                      regularizer=L2Decay(0.)),
                 name="fpn_{}".format(6))
        outs.append(fpn_blob)
        fpn_blob = fluid.layers.relu(fpn_blob)       
        fpn_blob_in = fpn_blob
        fpn_blob = fluid.layers.conv2d(
                  input=fpn_blob_in,
                  num_filters=out_channels,
                  filter_size=3,
                  stride=2,
                  padding=1,
                  param_attr=ParamAttr(name='fpn_{}_w'.format(7),
                                       initializer=Xavier()),
                  bias_attr=ParamAttr(name='fpn_{}_b'.format(7),     
                                       initializer=Constant(value=0.0),
                                       learning_rate=2.,
                                       regularizer=L2Decay(0.)),
                  name="fpn_{}".format(7))
        outs.append(fpn_blob)
    return tuple(outs)
