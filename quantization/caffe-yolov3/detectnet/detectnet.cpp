
/*
 * Company:	Systhesis
 * Author: 	Chen
 * Date:	2018/06/04	
 */

#include <stdio.h>
#include <signal.h>
#include <unistd.h>

#include "yolo_layer.h"
#include "image.h"
#include "cuda.h"

#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <sys/time.h>

#include <iostream>
#include <fstream>
#include <dirent.h>
#include "Markup.h"


using namespace caffe;
using namespace cv;
using namespace std;

// // quantization
// const string& model_file = "/home/user/Work/mjw/quantization/pytorch-caffe-darknet-convert/fast-yolov3.prototxt";
// const string& weights_file = "/home/user/Work/mjw/quantization/pytorch-caffe-darknet-convert/fast-yolov3.caffemodel";
// const string& model_file = "/home/user/Work/mjw/quantization/delete_bn/relu_yolov3_nobn.prototxt";
const string& weights_file = "/home/user/Work/mjw/quantization/delete_bn/relu_yolov3_nobn.caffemodel";
// const string& model_file = "/home/user/Work/mjw/quantization/delete_bn/fast-yolov3-carplate_nobn.prototxt";
const string& model_file = "/home/user/Work/mjw/quantization/ezai/deploy_quantization_concat.prototxt";
// const string& weights_file = "/home/user/Work/mjw/quantization/delete_bn/fast-yolov3-carplate_nobn.caffemodel";
const string rootdirPath = "/home/user/Work/mjw/quantization/QuanTestFiles2/";
const char* fi = "out.txt";
const char* fi_all = "out_all_results.txt";
const float a = 1.0;

// calculate IOU between two bboxs
float intersectRect(const cv::Rect& rectA, const cv::Rect& rectB){
    if (rectA.x > rectB.x + rectB.width) { return 0.; }
    if (rectA.y > rectB.y + rectB.height) { return 0.; }
    if ((rectA.x + rectA.width) < rectB.x) { return 0.; }
    if ((rectA.y + rectA.height) < rectB.y) { return 0.; }
    float colInt = min(rectA.x + rectA.width, rectB.x + rectB.width) - max(rectA.x, rectB.x);
    float rowInt = min(rectA.y + rectA.height, rectB.y + rectB.height) - max(rectA.y, rectB.y);
    float intersection = colInt * rowInt;
    float areaA = rectA.width * rectA.height;
    float areaB = rectB.width * rectB.height;
    float intersectionPercent = intersection / (areaA + areaB - intersection);
    return intersectionPercent;
}

uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}

void SplitString(const string& s, vector<string>& v, const char& c)
{
    std::string temp = "";
    for(int i = 0; i < (int)s.length();i++){
        if(s[i] != c) temp += s[i];
        else{
            v.push_back(temp);
            temp = "";
        }
    }
}

int main( int argc, char** argv )
{
    // Initialize the network.
    Caffe::set_mode(Caffe::GPU);

    /* Load the network. */
    boost::shared_ptr<Net<float>> net;
    net.reset(new Net<float>(model_file, TEST));
    net->CopyTrainedLayersFrom(weights_file);

    // printf("num_inputs is %d\n",net->num_inputs());
    // printf("num_outputs is %d\n",net->num_outputs());
    CHECK_EQ(net->num_inputs(), 1) << "Network should have exactly one input.";
    // CHECK_EQ(net->num_outputs(), 3) << "Network should have exactly three outputs.";	

    Blob<float> *input_data_blobs = net->input_blobs()[0];
    // LOG(INFO) << "Input data layer channels is  " << input_data_blobs->channels();
    // LOG(INFO) << "Input data layer width is  " << input_data_blobs->width();
    // LOG(INFO) << "Input data layer height is  " << input_data_blobs->height();

    int size = input_data_blobs->channels()*input_data_blobs->width()*input_data_blobs->height();
    DIR * dir;
    struct dirent * ptr;
    string xmlName,dirPath,copydirPath,imgPath;
    dir = opendir((char *)rootdirPath.c_str()); //打开一个目录
    int in_xml = 0;
    int notin_xml = 0;
    int notin_det = 0;
    uint64_t beginDetectTime =  current_timestamp();
    while((ptr = readdir(dir)) != NULL) //循环读取目录数据
    {
        // printf("d_name : %s\n", ptr->d_name); //输出文件名
        xmlName=ptr->d_name;
        char *strxml=".txt"; // change the dir file to txt
        char *strjpg=".jpg";
        char *strpng=".png";
        if (strstr(xmlName.c_str(), strjpg) != NULL || strstr(xmlName.c_str(), strpng) != NULL) // jpg or png
        {
            imgPath = rootdirPath + xmlName;
            // dirPath = rootdirPath + xmlName;
            copydirPath = imgPath;
            if (strstr(xmlName.c_str(), strjpg) != NULL)
            {
                dirPath = copydirPath.replace(copydirPath.find(".jpg"),4,".txt");
            }else{
                dirPath = copydirPath.replace(copydirPath.find(".png"),4,".txt");
            }
            
            printf("txt_name : %s\n", dirPath.c_str()); //输出txt文件绝对路径
            //read corresponding jpg
            // imgPath = copydirPath.replace(copydirPath.find(".txt"),4,".jpg");
            const char* imgPathChar = imgPath.c_str();
            Mat img = imread((char*)imgPathChar);
            int imgwidth=img.cols;
            int imgheight=img.rows;            
            int xml_xmin;
            int xml_ymin;
            int xml_xmax;
            int xml_ymax;
            int xml_width;
            int xml_height;
            vector<cv::Rect> xml_boxes;
            xml_boxes.clear();
            // // read corresponding xml
            // CMarkup xml_CMarkup;//声明xml cmark对象
            // xml_CMarkup.Load(dirPath);
            // xml_CMarkup.ResetMainPos(); //将当前主位置复位为第一个兄弟位置之前
            // while(xml_CMarkup.FindChildElem("object"))   //定位到下一个子元素，匹配元素名或路径
            // {
            //     xml_CMarkup.IntoElem();//进入当前主位置的下一级，当前的位置变为父位置
            //     // CString strTagName = _T("");
            //     // CString strData = _T("");
            //     // strTagName = xml.GetTagName();  //得到主位置元素（或正在进行的指令的）标签名称
            //     // strData = xml.GetData(); // 得到当前主位置元素或节点的字符串值
            //     xml_CMarkup.FindChildElem("bndbox");
            //     xml_CMarkup.IntoElem(); 
            //     xml_CMarkup.FindChildElem("xmin");
            //     string xml_xmin_str= xml_CMarkup.GetChildData();
            //     xml_CMarkup.FindChildElem("ymin");
            //     string xml_ymin_str= xml_CMarkup.GetChildData();
            //     xml_CMarkup.FindChildElem("xmax");
            //     string xml_xmax_str= xml_CMarkup.GetChildData();
            //     xml_CMarkup.FindChildElem("ymax");
            //     string xml_ymax_str= xml_CMarkup.GetChildData();
            //     // printf("xml_box : %s\t%s\t%s\t%s\n",xml_xmin_str.c_str(),xml_ymin_str.c_str(),xml_xmax_str.c_str(),xml_ymax_str.c_str());
            //     xml_xmin = atoi(xml_xmin_str.c_str());
            //     xml_ymin = atoi(xml_ymin_str.c_str());
            //     xml_xmax = atoi(xml_xmax_str.c_str());
            //     xml_ymax = atoi(xml_ymax_str.c_str());
            //     xml_width = xml_xmax-xml_xmin;
            //     xml_height = xml_ymax-xml_ymin;
            //     cv::Rect recxml(xml_xmin,xml_ymin,xml_width,xml_height);
            //     xml_boxes.push_back(recxml);
            //     xml_CMarkup.OutOfElem();
            //     xml_CMarkup.OutOfElem();
            // }
            // read corresponding txt
            std::ifstream fin(dirPath.c_str(), std::ios::in);
            char line[1024]={0};
            std::string id = "";
            std::string x = "";
            std::string y = "";
            std::string w = "";
            std::string h = "";
            // printf("%s\n",dirPath.c_str() );
            while(fin.getline(line, sizeof(line)))
            {
                // printf("%s\n", line);
                std::stringstream word(line);
                word >> id;
                word >> x;
                word >> y;
                word >> w;
                word >> h;
                xml_xmin = int((atof(x.c_str())*2-atof(w.c_str()))/2*imgwidth);
                xml_ymin = int((atof(y.c_str())*2-atof(h.c_str()))/2*imgheight);
                xml_width = int(atof(w.c_str())*imgwidth);
                xml_height = int(atof(w.c_str())*imgheight);
                cv::Rect rectxt(xml_xmin,xml_ymin,xml_width,xml_height);//top-left,width-height
                xml_boxes.push_back(rectxt);
            }
            fin.close();
            // imgPath = dirPath.replace(dirPath.find(".xml"),4,".jpg");
            // const char* imgPathChar = imgPath.c_str();
            // printf("img_name : %s\n", imgPath.c_str());//输出jpg文件绝对路径

            // uint64_t beginDataTime =  current_timestamp();
            //load image
            image im = load_image_color((char*)imgPathChar,0,0); // change channel order and rgb to bgr
            image sized = letterbox_image(im,input_data_blobs->width(),input_data_blobs->height()); // reseize image
            // for (int i = 184000; i < 185000; ++i)
            // {
            //     printf("%f",sized.data[i]);
            //     printf("%s", ",");
            // }
            
            cuda_push_array(input_data_blobs->mutable_gpu_data(),sized.data,size);
            // uint64_t endDataTime =  current_timestamp();

            //YOLOV3 objection detection implementation with Caffe
            // uint64_t beginDetectTime =  current_timestamp();

            net->Forward();

            vector<Blob<float>*> blobs;
            blobs.clear();
            // 3 scales
            Blob<float>* out_blob1 = net->output_blobs()[2];
            blobs.push_back(out_blob1);
            // printf("%d\n", net->output_blobs().size());
            Blob<float>* out_blob2 = net->output_blobs()[1];
            blobs.push_back(out_blob2);
            Blob<float>* out_blob3 = net->output_blobs()[0];
            blobs.push_back(out_blob3);

            // printf("output blob1 shape c= %d, h = %d, w = %d\n",out_blob1->channels(),out_blob1->height(),out_blob1->width());
            // printf("output blob2 shape c= %d, h = %d, w = %d\n",out_blob2->channels(),out_blob2->height(),out_blob2->width());
            int nboxes = 0;
            detection *dets = get_detections(blobs,im.w,im.h,&nboxes);

            // uint64_t endDetectTime = current_timestamp();
            // printf("object-detection:  finished processing data operation  (%zu)ms\n", endDataTime - beginDataTime);
            // printf("object-detection:  finished processing yolov3-tiny network  (%zu)ms\n", endDetectTime - beginDetectTime);

            //show detection results
            // Mat img = imread((char*)imgPathChar);
            // imgwidth=img.cols;
            // imgheight=img.rows;

            int i,j;
            int flags = 0;
            vector<cv::Rect> det_boxes;
            det_boxes.clear();
            for(i=0;i< nboxes;++i){
                int cls = -1;
                for(j=0;j<1;++j){
                    if(dets[i].prob[j] > 0){
                        if(cls < 0){
                            cls = j;
                        }
                        // printf("%d: %.0f%%\n",cls,dets[i].prob[j]*100);
                    }
                }
                if(cls >= 0){
                    box b = dets[i].bbox;
                    
                    int left  = (b.x-b.w/2.)*im.w;
                    int right = (b.x+b.w/2.)*im.w;
                    int top   = (b.y-b.h/2.)*im.h;
                    int bot   = (b.y+b.h/2.)*im.h;
                    // printf("(%d, %d, %d, %d, %f, %d)\n",left,right,top,bot,dets[i].prob[cls],cls);
                    if (cls==0) //car
                    {
                        rectangle(img,Point(left,top),Point(right,bot),Scalar(255,255,0),3,8,0);//yellow
                        cv::Point p = cv::Point(left,top); //加上字符的起始点
                        cv::putText(img, "car", p, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(160, 32, 240), 2, CV_AA); 
                        //在图像上加字符 //第一个参数为要加字符的目标函数 //第二个参数为要加的字符 //第三个参数为字体 //第四个参数为子的粗细 //第五个参数为字符的颜色

                    }else if (cls==1) //pedestrain
                    {
                        rectangle(img,Point(left,top),Point(right,bot),Scalar(0,255,0),3,8,0); //green
                        cv::Point p = cv::Point(left,top); //加上字符的起始点
                        cv::putText(img, "pedestrain", p, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(160, 32, 240), 2, CV_AA); 
                    }else{ //carplate
                        rectangle(img,Point(left,top),Point(right,bot),Scalar(255,0,0),3,8,0); //red
                        cv::Point p = cv::Point(left,top); //加上字符的起始点
                        cv::putText(img, "carplate", p, cv::FONT_HERSHEY_TRIPLEX, 0.8, cv::Scalar(160, 32, 240), 2, CV_AA); 
                    }
                    cv::Rect recdet(left,top,right-left,bot-top);
                    det_boxes.push_back(recdet);
                }
            }
            for (int i = 0; i < det_boxes.size(); ++i) {
                int flags =0;
                for (int j = 0; j < xml_boxes.size(); ++j) {
                    if (intersectRect(xml_boxes[j],det_boxes[i])>=0.5)
                    {
                        flags++;
                    }
                }
                if (flags >= 1)
                {
                    in_xml++;
                }else{
                    notin_xml++;
                }
            }
            for (int i = 0; i < xml_boxes.size(); ++i) {
                int flags = 0;
                for (int j = 0; j < det_boxes.size(); ++j) {
                    if (intersectRect(xml_boxes[i],det_boxes[j])>=0.5)
                    {
                        flags++;
                    }
                }
                if (flags != 1)
                {
                    notin_det++;
                }
            }
            // float precision = float(in_xml)/(in_xml+notin_xml);
            // float recall = float(in_xml)/(in_xml+notin_det);
            // float f1_score = (a*a+1)*precision*recall/(a*a*precision+recall);
            // printf("in_xml:%d,notin_xml:%d,notin_det:%d\n", in_xml,notin_xml,notin_det);
            // printf("precision:%f\n", precision);
            // printf("recall:%f\n", recall);
            // printf("f1_score:%f\n", f1_score);

            
            namedWindow("show",CV_WINDOW_NORMAL);
            resizeWindow("show",720, 480);
            imshow("show",img);
            waitKey(0);

            free_detections(dets,nboxes);
            free_image(im);
            free_image(sized);
        }
    }
    closedir(dir);//关闭目录指针

    float precision = float(in_xml)/(in_xml+notin_xml);
    float recall = float(in_xml)/(in_xml+notin_det);
    float f1_score = (a*a+1)*precision*recall/(a*a*precision+recall);
    printf("in_xml:%d,notin_xml:%d,notin_det:%d\n", in_xml,notin_xml,notin_det);
    printf("precision:%f\n", precision);
    printf("recall:%f\n", recall);
    printf("f1_score:%f\n", f1_score);

    // uint64_t endDetectTime = current_timestamp();
    // printf("Total Detection Time:  (%zu)ms\n", endDetectTime - beginDetectTime);

    // ofstream fi_outfile(fi);
    // ofstream fi_all_outfile(fi_all, ios::app);
    // fi_outfile << to_string(f1_score);
    // string temp = to_string(f1_score)+" in_xml="+to_string(in_xml)+" notin_xml="+to_string(notin_xml)+" notin_det="+to_string(notin_det)+" precision="+to_string(precision)+" recall="+to_string(recall);
    // fi_all_outfile << temp << endl;
    // fi_outfile.close();
    // fi_all_outfile.close();
    return 0;
}

