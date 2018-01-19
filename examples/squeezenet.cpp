// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "net.h"
#include <functional>
#include <zupply.hpp>

#include <iostream> 
#include <fstream> 
#include <Windows.h>  
using namespace zz;


inline int argmax(const std::vector<float>& arr) {
	return std::distance(arr.begin(), std::max_element(arr.begin(), arr.end()));
}

float sigmoid(float x)
{
	return (exp(x));
}

void readTxt(std::string file)
{
	std::ifstream infile;
	infile.open(file.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
	std::string s;
	while (getline(infile, s))
	{
		std::cout << s << std::endl;
	}
	infile.close();             //关闭文件输入流 
}


static int detect_squeezenet(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net squeezenet;
    squeezenet.load_param("D:/tools/ncnn2/ncnn/build/examples/model/ncnndong.param");
    squeezenet.load_model("D:/tools/ncnn2/ncnn/build/examples/model/ncnndong.bin");

	int width = 128;
	int height = 128;
    //ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows, width, height);
	//ncnn::Mat::PIXEL_RGBA2BGR
	//cv:: Mat face_gray, face_3channels;
	//ncnn::Mat in = ncnn::Mat::from_pixels(in_s.data, ncnn::Mat::PIXEL_RGBA2BGR, width, height);
	//cvtColor(face_gray, face_3channels, COLOR_GRAY2BGR);
	
	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2GRAY, bgr.cols, bgr.rows, width, height);
	
	//const float mean_vals[3] = {0.f, 0.f, 0.f};
	//const float mean_vals[3] = {104.f, 117.f, 123.f};
	//const float mean_vals[3] = { 123, 117.f, 104.f };
	const float  mean_vals[1] = { 127.5f };
	const float normal[1] = {0.0078125f};

    in.substract_mean_normalize(mean_vals, normal);	
    ncnn::Extractor ex = squeezenet.create_extractor();
	ex.set_num_threads(1);
    ex.set_light_mode(true);
    //ex.input(ncnn_proto_id::BLOB_data, in);
    //ncnn::Mat out;
   // ex.extract(ncnn_proto_id::BLOB_softmax, out);
	ex.input("data", in);
	ncnn::Mat out;
	//ex.extract("global_pool", out);
	//std::string layer_name = "pooling0";
	//ex.extract("pooling0", out);
	ex.extract("cls_prob", out);
    
	cls_scores.resize(out.c);
	std::cout << out.c << std::endl;
	float sum = 0;
 //   for (int j=0; j<out.c; j++)
 //   {
 //     const float* prob = out.data + out.cstep * j;
	//	sum += exp(prob[0]);
	//	std::cout <<j<<" prob[0]"<< prob[0] << std::endl;
 //   }
	//float part;
	//for (int j = 0; j<out.c; j++)
	//{
	//	const float* prob = out.data + out.cstep * j;
	//	cls_scores[j] = exp(prob[0])/sum;
	//	part = cls_scores[j]; 
	//	std::cout << j << "part" << part << std::endl;
	//}
	for (int j = 0; j<out.c; j++)
	{		
		const float* prob = out.data + out.cstep * j;
		sum += exp(prob[0]);
	}

	for (int j = 0; j<out.c; j++)
	{
		const float* prob = out.data + out.cstep * j;
		cls_scores[j] = prob[0];

		//cls_scores[j] = exp(prob[0]) / sum;
		//std::cout << j <<"==="<< prob[0] << std::endl;
//		std::cout << j << "part" << cls_scores[j] << std::endl;
	}
    return 0;
}


static int print_topk(const std::vector<float>& cls_scores, int topk)
{
    // partial sort topk with index
    int size = cls_scores.size();
    std::vector< std::pair<float, int> > vec;
    vec.resize(size);
    for (int i=0; i<size; i++)
    {
        vec[i] = std::make_pair(cls_scores[i], i);
    }
    std::partial_sort(vec.begin(), vec.begin() + topk, vec.end(),
                      std::greater< std::pair<float, int> >());

    // print topk and score
    for (int i=0; i<topk; i++)
    {
        float score = vec[i].first;
        int index = vec[i].second;
        fprintf(stderr, "%d = %f\n", index, score);
    }
    return vec[0].second;
}

int main(int argc, char** argv)
{
	
	std::string path  = "./data/Path5.txt";
	std::ifstream infile;
	infile.open(path.data());   //将文件流对象与文件连接起来 
	assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
	std::string s;
	int count = 0;
	while (getline(infile, s))
	{
		if (infile.eof())  return -1;
		std::cout << s << std::endl;
		std::string imagepath = s;
		cv::Mat face = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
		cv::Mat face_gray, face_3channels;
		face_3channels = face;
		if (face_3channels.empty())
		{
			fprintf(stderr, "cv::imread %s failed\n", imagepath);
			return -1;
		}
		time::Timer timer;
		std::vector<float> cls_scores;
		detect_squeezenet(face_3channels, cls_scores);
		char tmp[32];
		sprintf(tmp, "%.3f", cls_scores[1]);
		std::string result_str = std::string("ssss") + "cls_scores= " + tmp;
		//logger->info("Gender forward elapsed time: ") << timer.to_string();
		int label = print_topk(cls_scores, 2);
		cv::imshow("img", face);
		cv::waitKey(0);
		if (label == 1) {
			count++;
		}
		//cv::imshow("img", face);
		//cv::waitKey(0);
		//system("pause");
		}	
	fprintf(stderr, "right = %d", count);
	}



	//auto logger = log::get_logger("default");
 //   const char* imagepath = argv[1];
	////imagepath = "s1.jpg";
	//imagepath = "data/a6.jpg";
	//	
 //   cv::Mat face = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
	////for (int h = 0; h < face.rows; h++) {
	////	for (int w = 0; w < face.cols; w++) {
	////		int pixel_1 = face.at<cv::Vec3b>(h, w)[0];
	////		int pixel_2 = face.at<cv::Vec3b>(h, w)[0];
	////		int pixel_3 = face.at<cv::Vec3b>(h, w)[0];

	////		fprintf(stderr, "%d %d %d \n", pixel_1, pixel_2, pixel_3);
	////	}
	////}
	//cv::Mat face_gray, face_3channels;
	////cv::imshow("img", face);
	////cvWaitKey(10);
	////cv::cvtColor(face, face_gray, cv::COLOR_BGR2GRAY);
	////cv::cvtColor(face_gray, face_3channels, cv::COLOR_GRAY2BGR);
	//face_3channels = face;	
	////cv::resize(face, face_3channels, cv::Size(192, 192));
 //   if (face_3channels.empty())
 //   {
 //       fprintf(stderr, "cv::imread %s failed\n", imagepath);
 //       return -1;
 //   }
	//time::Timer timer;
 //   std::vector<float> cls_scores;
 //   detect_squeezenet(face_3channels, cls_scores);
	//char tmp[32];
	//sprintf(tmp, "%.3f", cls_scores[1]);
	//std::string result_str = std::string("ssss")+"cls_scores= " + tmp;
	//logger->info("Gender forward elapsed time: ") << timer.to_string();
	//print_topk(cls_scores, 2);
	//system("pause");
//}

//int main()
//{
//	cv::VideoCapture cap(0);
//	while (1) {
//		cv::Mat frame;
//		bool ret = cap.read(frame);
//		if (ret) {
//			cv::Mat face = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
//			cv::Mat face_gray, face_3channels;
//			cv::cvtColor(face, face_gray, cv::COLOR_BGR2GRAY);
//			cv::cvtColor(face_gray, face_3channels, cv::COLOR_GRAY2BGR);
//		}
//	}
//	imagepath = "bin.jpg";
//    cv::Mat face = cv::imread(imagepath, CV_LOAD_IMAGE_COLOR);
//	cv::Mat face_gray, face_3channels;
//	cv::cvtColor(face, face_gray, cv::COLOR_BGR2GRAY);
//	cv::cvtColor(face_gray, face_3channels, cv::COLOR_GRAY2BGR);
//	face_3channels = face;
//    if (face_3channels.empty())
//    {
//        fprintf(stderr, "cv::imread %s failed\n", imagepath);
//        return -1;
//    }
//    std::vector<float> cls_scores;
//    detect_squeezenet(face_3channels, cls_scores);
//    print_topk(cls_scores, 2);
//}
