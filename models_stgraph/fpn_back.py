import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from config import cfg


def add_fpn_neck(inputs, 
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=-1,
                 ):
    
    if end_level == -1:
        end_level = len(inputs)

    # build laterals
    lateral_convs = []
    print("end_level: {}".format(end_level))
    for i in range(start_level, end_level):
        conv = fluid.layers.conv2d(
            input=inputs[i],
            num_filters=out_channels,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            param_attr=ParamAttr(name="res{}_lateral_w".format(i+3+start_level), 
                                 initializer=fluid.initializer.Xavier()),
            bias_attr=ParamAttr(name="res{}_lateral_b".format(i+3+start_level), 
                                 initializer=fluid.initializer.Constant(value=0.0),
                                 learning_rate=2.,
                                 regularizer=L2Decay(0.)),
            name="res{}_lateral".format(i+3+start_level))
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
    print("used_body_levels: {}".format(used_body_levels))
    for i in range(used_body_levels - 1, 0, -1):
        lateral_shape = fluid.layers.shape(lateral_convs[i - 1])
        lateral_shape = fluid.layers.slice(lateral_shape, axes=[0], starts=[2], ends=[4])
        lateral_shape.stop_gradient = True
        top_down = fluid.layers.resize_nearest(lateral_convs[i], 
                                               out_shape=[48,48],
                                               scale=2.0,
                                               actual_shape=lateral_shape,
                                               name="res{}_top-down".format(i+3+start_level))
        lateral_convs[i - 1] = fluid.layers.elementwise_add(
            x=lateral_convs[i - 1], 
            y= top_down
            )
        
    # build outputs
    # part 1: from original levels
    outs = []
    for i in range(used_body_levels):
        conv = fluid.layers.conv2d(
            input=lateral_convs[i],
            num_filters=out_channels,
            filter_size=3,
            stride=1,
            padding=1,
            act=None,
            param_attr=ParamAttr(name="fpn{}_w".format(i), 
                                 initializer=fluid.initializer.Xavier()),
            bias_attr=ParamAttr(name="fpn{}_b".format(i), 
                                 initializer=fluid.initializer.Constant(value=0.0),
                                 learning_rate=2.,
                                 regularizer=L2Decay(0.)),
            name="fpn{}".format(i))
        outs.append(conv)

    # part 2: add extra levels
    if(num_outs > used_body_levels):
        fpn_blob = inputs[end_level - 1]
        for i in range(end_level, num_outs):
            fpn_blob_in = fpn_blob
            fpn_blob = fluid.layers.conv2d(
                input=fpn_blob_in,
                num_filters=out_channels,
                filter_size=3,
                stride=2,
                padding=1,
                act='relu',
                param_attr=ParamAttr(name="fpn{}_w".format(i), 
                                     initializer=fluid.initializer.Xavier()),
                bias_attr=ParamAttr(name="fpn{}_b".format(i), 
                                     initializer=fluid.initializer.Constant(value=0.0),
                                     learning_rate=2.,
                                     regularizer=L2Decay(0.)),
                name="fpn{}".format(i))
            outs.append(conv)


    return tuple(outs)
