import os
import sys
import time
import shutil

net_src = "/home/user/Work/mjw/quantization/pytorch-caffe-darknet-convert/relu_yolov3.prototxt"  #src_prototxt
model_src = "/home/user/Work/mjw/quantization/pytorch-caffe-darknet-convert/relu_yolov3.caffemodel"  #32bits caffemodel
model_bnscale011 = "/home/user/Work/mjw/quantization/delete_bn/relu_yolov3_nobn.caffemodel"  #bnscale011 caffemodel

os.system("python2 auto_bnscale011.py --net0="+net_src+ \
	" --model="+model_src+" --model_bn="+model_bnscale011)