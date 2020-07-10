# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import sys
import cv2
# from pascal_voc_writer import Writer
#import matplotlib.pyplot as plt
# display plots in this notebook
import argparse
# set display defaults
#plt.rcParams['figure.figsize'] = (10, 10)        # large images
#plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
#plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

# The caffe module needs to be on the Python path;
#  we'll add it here explicitly.
import sys
caffe_root = '/home/user/Work/mjw/test/ezai/caffe_base_ezai/'  # this file should be run from {caffe_root}/examples (otherwise change this line)
sys.path.insert(0, caffe_root + 'python')
import os
import caffe
import math
from os import walk
from os.path import join

CLASSES = ('__background__','carplate')
fi = "out.txt"
fi_all = "out_all_results.txt"
in_xml = 0
notin_xml = 0
notin_det = 0
a = 1.0
image_dir='/home/user/Work/mjw/test/ezai/hand/hand_detection_test/'
# model_def='/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/quantization/deploy_quantization.prototxt'
# model_weights='/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/quantization/yolo-tiny_nobn.caffemodel'
model_def = '/home/user/Work/mjw/test/ezai/hand/hand_detection_quantization/deploy_quantization_concat_det.prototxt'
model_weights = '/home/user/Work/mjw/test/ezai/hand/hand_detection_quantization/yolov3-tiny-ZHhands-delete-bn.caffemodel'
# model_def = '/home/user/Work/mjw/experiment/MobileNet-YOLO/models/darknet_yolov3/relu-qi/yolov3-tiny.prototxt'
# model_weights = '/home/user/Work/mjw/experiment/MobileNet-YOLO/models/darknet_yolov3/relu-qi/yolov3-tiny_bn011.caffemodel'

# model_def = '/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/quantization/relu-qi/yolov3-tiny.prototxt'
# model_weights = '/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/quantization/relu-qi/yolov3-tiny.caffemodel'
image_resize=416


# calculate IOU between two bboxs
def calcIOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    a_x_center = float(boxA[0] + boxA[2])/2.0
    a_y_center = float(boxA[1] + boxA[3])/2.0
    a_width = float(boxA[2] - boxA[0])
    a_height = float(boxA[3] - boxA[1])
    b_x_center = float(boxB[0] + boxB[2])/2.0
    b_y_center = float(boxB[1] + boxB[3])/2.0
    b_width = float(boxB[2] - boxB[0])
    b_height = float(boxB[3] - boxB[1])
    if((abs(a_x_center - b_x_center) < ((a_width + b_width) / 2.0)) and (abs(a_y_center - b_y_center) < ((a_height + b_height) / 2.0))):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
     
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
     
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
     
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
    else:
        iou = 0
    # return the intersection over union value
    return iou

def vis_detections(image,result) :
    w = image.shape[1]
    h = image.shape[0]
    for i in range(result.shape[1]):
        left = result[0][i][3] * w
        top = result[0][i][4] * h
        right = result[0][i][5] * w
        bot = result[0][i][6] * h
        score = result[0][i][2]
        label = result[0][i][1]
        if(score>0.5) :
            print(left,right,top,bot,score,label)
            cv2.rectangle(image,(int(left), int(top)),(int(right),int(bot)),(0,255,0), 2)

            label = '{:s} {:.3f}'.format(CLASSES[int(label)], score)
            font = cv2.FONT_HERSHEY_SIMPLEX
            size = cv2.getTextSize(label, font, 0.5, 0)[0]
            cv2.rectangle(image,(int(left), int(top)),
                    (int(left+size[0]),int(top+ size[1])),(0,255,0), -1)

            cv2.putText(image, label,(int(left+0.5), int(top+ size[1]+0.5)),font,0.5,(0,0,0),0)
def det(image,transformer,net):
    
    transformed_image = transformer.preprocess('data', image)
    #plt.imshow(image)

    net.blobs['data'].data[...] = transformed_image

    ### perform classification
    output = net.forward()

    res = output['layer24-yolo'][0]  # the output probability vector for the first image in the batch
    print(res.shape)
    # print(res.shape[1])
    return res

def convert_txt_xml(image,pic_txt):
    w = image.shape[1]
    h = image.shape[0]
    voc_boxes = []
    txt = open(pic_txt, 'r')
    for line in txt.readlines():
        line = line.strip()
        yolo_center_x   = round(float(line.split(' ')[1]) * w)
        yolo_center_y   = round(float(line.split(' ')[2]) * h)
        yolo_obj_width  = round(float(line.split(' ')[3]) * w)
        yolo_obj_height = round(float(line.split(' ')[4]) * h)
        voc_xmin = int(yolo_center_x - yolo_obj_width/2)
        voc_ymin = int(yolo_center_y - yolo_obj_height/2)
        voc_xmax = int(yolo_center_x + yolo_obj_width/2)
        voc_ymax = int(yolo_center_y + yolo_obj_height/2)
        voc_boxes.append([voc_xmin,voc_ymin,voc_xmax,voc_ymax])
        pass
    return voc_boxes

def convert_detections(image,result):
    w = image.shape[1]
    h = image.shape[0]
    det_boxes = []
    for i in range(result.shape[1]):
        left = result[0][i][3] * w
        top = result[0][i][4] * h
        right = result[0][i][5] * w
        bot = result[0][i][6] * h
        score = result[0][i][2]
        label = result[0][i][1]
        if(score>0.5) :
            det_boxes.append([int(left),int(top),int(right),int(bot)])
    return det_boxes

def is_imag(filename):
    return filename[-4:] in ['.png', '.jpg']

# def main(args):    
caffe.set_mode_cpu()
# caffe.set_mode_gpu()
# caffe.set_device(3)
# model_def = args.model_def
# model_weights = args.model_weights

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
mu = np.array([0,0,0])
# mu = np.array([104, 117, 123])
# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 1.0)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          image_resize, image_resize) 

filenames = os.listdir(image_dir)
images = filter(is_imag, filenames)
for image in images :
    pic = image_dir + image
    print pic
    pic_txt = image_dir + image.split('.')[0] + '.txt'
    # input = caffe.io.load_image(pic)
    image_show =cv2.imread(pic)  
    result = det(image_show/255.0,transformer,net)
    # print result
    xml_boxes = []
    # xml_boxes = convert_txt_xml(image_show,pic_txt)
    det_boxes = convert_detections(image_show,result)
    vis_detections(image_show,result)
    cv2.imshow("Image", image_show)
    cv2.waitKey (0)
    for det_box in det_boxes:
        flags=0
        for xml_box in xml_boxes:
            if calcIOU(det_box,xml_box)>=0.5:
                flags+=1
                pass
            pass
        if flags >= 1:
            in_xml+=1
        else:
            notin_xml+=1
        pass
    for xml_box in xml_boxes:
        flags = 0
        for det_box in det_boxes:
            if calcIOU(det_box,xml_box)>=0.5:
                flags+=1
                pass
            pass
        if flags!=1:
            notin_det+=1
        pass
if in_xml==0:
    precision = 0
    recall = 0
    f1_score = 0
else:
    precision = float(in_xml)/(in_xml+notin_xml)
    recall = float(in_xml)/(in_xml+notin_det)
    f1_score = (a*a+1)*precision*recall/(a*a*precision+recall)
print("in_xml:%d,notin_xml:%d,notin_det:%d"%(in_xml,notin_xml,notin_det))
print("precision:%f"%precision)
print("recall:%f"%recall)
print("f1_score:%f"%f1_score)

fn = open(fi,"w")
fn_all = open(fi_all,"a+")
fn.write(str(f1_score))
temp = str(f1_score)+" in_xml="+str(in_xml)+" notin_xml="+str(notin_xml)+" notin_det="+str(notin_det)+" precision="+str(precision)+" recall="+str(recall)
fn_all.write(temp+"\n")
fn.close()
fn_all.close()
        
# def parse_args():
#     parser = argparse.ArgumentParser()
#     '''parse args'''
#     parser.add_argument('--image_dir', default='/home/user/Work/mjw/yolo-master/CarplateInCar/data/')
#     # parser.add_argument('--image_dir', default='/home/user/Work/mjw/yolo-master/darknet/data/voc/VOCdevkit/VOC2007/JPEGImages/')
#     # parser.add_argument('--image_dir', default='/home/user/Work/mjw/yolo-master/darknet/data/coco/images/val2014/')
#     parser.add_argument('--model_def', default='/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/yolo-tiny_nobn.prototxt')
#     parser.add_argument('--model_weights', default='/home/user/Work/mjw/test/ezai/all_projects/yolo/models/caffe/CarplateInCar/yolo-tiny_nobn.caffemodel')
#     # parser.add_argument('--model_def', default='models/darknet_yolov3/yolov3-spp.prototxt')
#     # parser.add_argument('--model_weights', default='models/darknet_yolov3/yolov3-spp-iter-1000.caffemodel')
#     parser.add_argument('--image_resize', default=416, type=int)
#     # parser.add_argument('--write_voc', default=False)
#     return parser.parse_args()
    
# if __name__ == '__main__':
#     main(parse_args())