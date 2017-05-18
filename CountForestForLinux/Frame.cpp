#include "Frame.h"
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
namespace CrowdCount {
	using std::cout;
	using std::endl;
	Frame::Frame(std::string path, cv::Mat pmap,cv::Mat roi)
	{
		
		originImg_ = cv::imread(path,0);
		
		cols_ = originImg_.cols;
		rows_ = originImg_.rows;
		
		log_ = cv::Mat(rows_, cols_, CV_32F);
		
		structVal_ = cv::Mat(rows_, cols_, CV_32FC2);

		lbp_ = cv::Mat(rows_, cols_, CV_8U);
		gg_ = cv::Mat(rows_, cols_, CV_32F);
		cv::Mat background;
		
		medianBlur(originImg_, background, 9);
		cv::Mat dst;
		
		absdiff(originImg_, background, dst);
		cv::Mat binaryImg;
		
		threshold(dst, binaryImg, 50, 1, CV_THRESH_BINARY);
		
		//std::cout << "leaf density (default) = " << std::endl << binaryImg << std::endl << std::endl;
		originImg_ = originImg_.mul(binaryImg);
		
		
		originImg_ = originImg_.mul(roi);
		
		
		GaussianBlur(originImg_, gaussianBlur_, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		Sobel(gaussianBlur_, dx, CV_32FC1, 1, 0);
		Sobel(gaussianBlur_, dy, CV_32FC1, 0, 1);
		
		multiply(dx, dx, dx2);
		multiply(dy, dy, dy2);
		multiply(dx, dy, dxy);
		
		getLog();
		getStructVal();
		getLbp();
		getGG();
	
		for (int r = 0; r < originImg_.rows; r++)
		{
			for (int c = 0; c < originImg_.cols; c++)
			{
				log_.at<float>(r, c) = log_.at<float>(r, c) * pmap.at<float>(r, c)*pmap.at<float>(r, c);
				structVal_.at<cv::Vec2f>(r, c)[0] = structVal_.at<cv::Vec2f>(r, c)[0] * pmap.at<float>(r, c)*pmap.at<float>(r, c);
				structVal_.at<cv::Vec2f>(r, c)[1] = structVal_.at<cv::Vec2f>(r, c)[1] * pmap.at<float>(r, c)*pmap.at<float>(r, c);
				lbp_.at<float>(r, c) = lbp_.at<float>(r, c) * pmap.at<float>(r, c)*pmap.at<float>(r, c);
				gg_.at<float>(r, c) = gg_.at<float>(r, c) * pmap.at<float>(r, c)*pmap.at<float>(r, c);
			}
		}


	}
	Frame::Frame(std::string path, cv::Mat densityMap) {
		this->densityMapPath = path;
		this->densityMap = densityMap;
	}


	Frame::~Frame()
	{
	}

}