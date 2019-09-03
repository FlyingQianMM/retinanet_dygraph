import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Constant
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.initializer import Xavier
from paddle.fluid.initializer import Normal
from config import cfg

class add_fpn_neck(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 out_channels=256,
                 num_outs=5,
                 start_level=0,
                 end_level=3):
        super(add_fpn_neck, self).__init__(name_scope)

        self.inputs_name = ['res3_3_sum', 'res4_5_sum', 'res5_2_sum']
        self.start_level = start_level
        self.end_level = end_level
        self.num_outs = num_outs

        self.lateral_conv_list = []
        for i in range(self.start_level, self.end_level):
            if i < (self.end_level-1):
                lateral_name = 'fpn_inner_' + self.inputs_name[i] + '_lateral'
            else:
                lateral_name = 'fpn_inner_' + self.inputs_name[i]
            lateral_conv = self.add_sublayer(
                lateral_name,
                Conv2D(lateral_name,
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
                                           regularizer=L2Decay(0.))
                       ))
            self.lateral_conv_list.append(lateral_conv)

        self.out_conv_list = []
        for i in range(self.end_level - self.start_level):
            fpn_name = 'fpn_' + self.inputs_name[i]
            out_conv = self.add_sublayer(
                fpn_name,
                Conv2D(lateral_name,
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
                                           regularizer=L2Decay(0.))
                       ))
            self.out_conv_list.append(out_conv)

        if(self.num_outs > (self.end_level - self.start_level)):
            self.out_conv6 = Conv2D("fpn_{}".format(6),
                num_filters=out_channels,
                filter_size=3,
                stride=2,
                padding=1,
                act=None,
                param_attr=ParamAttr(name='fpn_{}_w'.format(6),
                                     initializer=Xavier()),
                bias_attr=ParamAttr(name='fpn_{}_b'.format(6),
                                    initializer=Constant(value=0.0),
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.))
                )
            self.out_conv7 = Conv2D("fpn_{}".format(7),
                num_filters=out_channels,
                filter_size=3,
                stride=2,
                padding=1,
                act=None,
                param_attr=ParamAttr(name='fpn_{}_w'.format(7),
                                     initializer=Xavier()),
                bias_attr=ParamAttr(name='fpn_{}_b'.format(7),
                                    initializer=Constant(value=0.0),
                                    learning_rate=2.,
                                    regularizer=L2Decay(0.))
                )

    def forward(self, inputs):
        # build laterals
        lateral_convs = []
        for i in range(self.start_level, self.end_level):
            conv = self.lateral_conv_list[i](inputs[i])
            lateral_convs.append(conv)

        # build top-down path
        used_body_levels = len(lateral_convs)
        for i in range(used_body_levels - 1, 0, -1):
            shape = fluid.layers.shape(lateral_convs[i])
            shape._stop_gradient = False
            shape_gpu_data = shape.numpy()
            shape_gpu = to_variable(shape_gpu_data)
            shape_gpu._stop_gradient = False
            shape_hw = fluid.layers.slice(shape_gpu, axes=[0], starts=[2], ends=[4])
            shape_hw._stop_gradient = True
            in_shape = fluid.layers.cast(shape_hw, dtype='int32')
            in_shape._stop_gradient = True
            out_shape = in_shape * 2
            out_shape._stop_gradient = True
            fpn_inner_name = 'fpn_inner_' + self.inputs_name[i-1]
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
            conv = self.out_conv_list[i](lateral_convs[i])
            outs.append(conv)
        # part 2: add extra levels
        if(self.num_outs > used_body_levels):
            fpn_blob = inputs[self.end_level - 1]
            fpn_blob_in = fpn_blob
            fpn_blob = self.out_conv6(fpn_blob_in)
            outs.append(fpn_blob)
            fpn_blob = fluid.layers.relu(fpn_blob)
            fpn_blob_in = fpn_blob
            fpn_blob = self.out_conv7(fpn_blob_in)
            outs.append(fpn_blob)

        return tuple(outs)
