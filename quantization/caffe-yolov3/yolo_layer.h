/*
 * Company:	Systhesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_
#include <caffe/caffe.hpp>
#include <string>
#include <vector>

using namespace caffe;

const int netW = 608;
const int netH = 608;
const int classes = 1;//80
const float thresh = 0.5;//0.5
const float hier_thresh = 0.5;//0.5
const float nms = 0.45;
const int numBBoxes = 3;//3
const int numAnchors = 9;//9
const int relative = 1;

typedef struct{
    float x,y,w,h;
}box;

typedef struct{
    box bbox;
    int classes;
    float* prob;
    float* mask;
    float objectness;
    int sort_class;
}detection;

typedef struct layer{
    int batch;
    int total;
    int n,c,h,w;
    int out_n,out_c,out_h,out_w;
    int classes;
    int inputs,outputs;
    int *mask;
    float* biases;
    float* output;
    float* output_gpu;
}layer;

layer make_yolo_layer(int batch,int w,int h,int n,int total,int classes);

void free_yolo_layer(layer l);

void forward_yolo_layer_gpu(const float* input,layer l, float* output);

detection* get_detections(vector<Blob<float>*> blobs,int img_w,int img_h,int* nboxes);

void free_detections(detection *dets,int nboxes);




#endif