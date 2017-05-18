#include "Patch.h"
namespace CrowdCount {

	cv::Mat Patch::calHistogram(float pmapVal) {
		peopleCount = countNonZero(label);
		cv::Mat gaussian_blur;
		//cout << "label= " << endl << label << endl << endl;
		
		cv::Mat float_label;
		label.convertTo(float_label,CV_32F);

		cv::GaussianBlur(float_label, gaussian_blur, cv::Size(9, 9), 0.2*pmapVal);
		/*std::cout << gaussian_blur.type() << std::endl;*/
	/*	for (int r = 0; r < gaussian_blur.rows; r++) {
			for (int c = 0; c<gaussian_blur.cols; c++)
			{
				std::cout << gaussian_blur.at<float>(r, c) << "  ";
			} 
			std::cout<<std::endl;
		}*/

		/*cout << "gaussian_blur= " << endl << gaussian_blur << endl << endl;*/

		cv::Mat tmpresult;
		cv::normalize(gaussian_blur, tmpresult);    //高斯模糊后归一化矩阵元素到0-1之间
		cv::Mat result;
		cv::addWeighted(tmpresult, 31, tmpresult, 0, 0, result); //将矩阵元素放缩到0-255之间

		//cout << "result = " << endl << result << endl << endl;
		histogram = cv::Mat::zeros(32, 1, CV_8U);
		for (int r = 0; r < result.rows; r++) {
			for (int c = 0; c<result.cols; c++)
			{
				/*std::cout << unsigned(result.at<float>(r, c)) << "  ";*/
				histogram.at<uchar>(unsigned(result.at<float>(r, c)), 0)++;
			}
			//std::cout << std::endl;
		}
		//cout << "histogram = " << endl << histogram << endl << endl;
		return histogram;
	};

}
