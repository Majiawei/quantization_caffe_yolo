#!/usr/bin/env sh

./caffe_base_ezai/build/tools/ristretto quantize \
	--model=/home/user/Work/mjw/quantization/delete_bn/relu_yolov3_nobn.prototxt \
	--model_quantized=/home/user/Work/mjw/quantization/ezai/deploy_quantization.prototxt \
	--qt_script_name="/home/user/Work/mjw/quantization/yolov3_caffe_det/caffe-yolov3/build/x86_64/bin/detectnet" \
	--PerAccuracy=/home/user/Work/mjw/quantization/ezai/out.txt \
	--trimming_mode=dynamic_fixed_point \
	--gpu=3 \
	--fn_all_results=/home/user/Work/mjw/quantization/ezai/out_all_results.txt
