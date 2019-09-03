# Retinanet

## 安装

**动态图中尚未支持Linear lr warm up + Piecewise decay，因此需要自行添加并重新编译paddle。** 代码放在文件夹paddle中。

**安装[cocoapi](https://github.com/cocodataset/cocoapi)：**

训练前需要首先下载[cocoapi](https://github.com/cocodataset/cocoapi)：

    git clone https://github.com/cocodataset/cocoapi.git
    cd cocoapi/PythonAPI
    # if cython is not installed
    pip install Cython
    # Install into global site-packages
    make install
    # Alternatively, if you do not have permissions or prefer
    # not to install the COCO API into global site-packages
    python2 setup.py install --user

## 数据准备

在[MS-COCO数据集](http://cocodataset.org/#download)上进行训练，通过如下方式下载数据集。

    cd dataset/coco
    ./download.sh

## 模型训练
    
**编译安装cython实现的三个函数：**
```
python setup.py build_ext --inplace
```

**解压预训练模型**
```
tar -xvf imagenet_resnet50_fusebn.tar
```

**启动训练**
- 单卡训练：
    ```
    export CUDA_VISIBLE_DEVICES=0
    python train.py \
       --model_save_dir=output/ \
       --pretrained_model=imagenet_resnet50_fusebn \
       --data_dir=${path_to_data} \
       --use_data_parallel=0
    ```

- 多卡多线程训练

    ```
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir ./mylog train.py \
       --model_save_dir=output/ \
       --pretrained_model=imagenet_resnet50_fusebn \
       --data_dir=${path_to_data} \
       --use_data_parallel=1
    ```
    
## 模型评估
将动态图模型格式转换成静态图格式：
```
python convert_dygraph.py --model_dir=${path_to_model}
```

启动评估：
```
python eval_coco_map.py --pretrained_model=${path_to_model} --use_data_parallel 0 --data_dir=${path_to_data} 
```
