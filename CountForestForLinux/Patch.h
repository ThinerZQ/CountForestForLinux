#pragma once
#include <opencv2/opencv.hpp>

namespace CrowdCount {
	using namespace std;
	class Patch{
		
		public:
		std::vector<float> feature;
		cv::Mat label;
		int peopleCount;
		cv::Mat predictLabel;
		cv::Mat histogram;
		bool hasForeground;
		int patchcenterX;//row
		int patchcenterY; //col
		Patch() {};
		
		Patch(int x,int y):patchcenterX(x),patchcenterY(y){};
		cv::Mat calHistogram(float pmapVal);
	};
}