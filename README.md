# quantization_caffe_yolo
《训练》
.
└── 训练
    ├── extract_log.py
    ├── k-meansforv3.py
    ├── tiny.data
    ├── tiny.names
    ├── train_loss_visualization.py
    └── yolov3-tiny.cfg
1、k-meansforv3.py聚类生成6个anchor(修改cfg文件中anchor数及yolo前一层filter数)
2、训练语句(生成log文件)：
./darknet detector train yolov3.data yolov3.cfg -gpus 1,2,3 2>1 | tee train.log
3、extract_log.py提取log信息
4、train_loss_visualization.py绘制loss曲线
《转换》
参考https://github.com/ChenYingpeng/darknet2caffe
无需添加层，引用ezai/caffe_base_ezai/python
《量化》
└── 量化
    ├── auto_bnscale011.py
    ├── bnscale011_script.py
    ├── caffe-yolov3
    ├── quantization_autoyolocarface.sh
    └── quantization_runonce.sh
1、删除bn层
①执行python bnscale011_script.py,删除caffemodel的bn层
②手动删除prototxt的bn层
2、单线层模型测试caffe-yolov3
修改makefile指定caffe相关位置
①修改yolo_layer.h相关参数
②修改yolo_layer.cpp参数
anchor参数
层参数
③执行sh runonce.sh(编译运行)
3、多线层模型测试caffe-yolov3
将detectnet.cpp.multi改为detectnet.cpp重新编译运行
4、执行量化
修改quantization_autoyolocarface.sh
执行quantization_runonce.sh



