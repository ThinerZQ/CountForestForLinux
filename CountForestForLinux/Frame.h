#pragma once
#include <opencv2/opencv.hpp>
namespace CrowdCount {
	class Frame
	{
	public:
		unsigned int cols_;
		unsigned int rows_;
		cv::Mat originImg_;
		cv::Mat log_;
		cv::Mat structVal_;
		cv::Mat lbp_;
		cv::Mat gg_;
		cv::Mat dx;
		cv::Mat dy;
		cv::Mat dx2;
		cv::Mat dy2;
		cv::Mat dxy;
		cv::Mat gaussianBlur_;
		std::string densityMapPath;
		cv::Mat densityMap;
		Frame(std::string path, cv::Mat densityMap);
		Frame(std::string path, cv::Mat pmap,cv::Mat roi);
		~Frame();
	private:
		void getLog() {
			//计算Laplacian Guassian
			cv::Mat tmpimg;
			GaussianBlur(originImg_, tmpimg, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
			cv::Mat tempLog;
			Laplacian(tmpimg, tempLog, CV_16S, 3);
			tempLog.convertTo(log_, CV_32F);

			//std::cout << "gaussian Laplacian:" << log_.rows << "  " << log_.cols << std::endl;
		}
		void getStructVal() {
			//计算结构张量
			cv::Mat t(2, 2, CV_32F); // tensor matrix
			for (int c = 0; c < dx2.cols; c++)
			{
				for (int r = 0; r < dx2.rows; r++)
				{
					// Insert values to the tensor matrix.
					t.at<float>(0, 0) = dx2.at<float>(r, c);
					t.at<float>(0, 1) = dxy.at<float>(r, c);
					t.at<float>(1, 0) = dxy.at<float>(r, c);
					t.at<float>(1, 1) = dy2.at<float>(r, c);
					// eigen decomposition to get the main gradient direction. 
					cv::Mat eigVal, eigVec;
					eigen(t, eigVal, eigVec);
					structVal_.at<cv::Vec2f>(r, c) = eigVal;

				}
			}
			//std::cout << "structVal:" << structVal_.rows << " " << structVal_.cols << std::endl;

		}
		void getLbp() {
			//计算LBP算子
			elbp(originImg_, lbp_, 1, 8);
			cv::Mat tempLBP;
			lbp_.convertTo(tempLBP, CV_32F);
			lbp_ = tempLBP;
			//std::cout << "lbp:" << lbp_.rows << "  " << lbp_.cols << std::endl;
		}

		void elbp(cv::Mat& src, cv::Mat &dst, int radius, int neighbors)
		{

			for (int n = 0; n<neighbors; n++)
			{
				// 采样点的计算
				float x = static_cast<float>(-radius * sin(2.0*CV_PI*n / static_cast<float>(neighbors)));
				float y = static_cast<float>(radius * cos(2.0*CV_PI*n / static_cast<float>(neighbors)));
				// 上取整和下取整的值
				int fx = static_cast<int>(floor(x));
				int fy = static_cast<int>(floor(y));
				int cx = static_cast<int>(ceil(x));
				int cy = static_cast<int>(ceil(y));
				// 小数部分
				float ty = y - fy;
				float tx = x - fx;
				// 设置插值权重
				float w1 = (1 - tx) * (1 - ty);
				float w2 = tx  * (1 - ty);
				float w3 = (1 - tx) *      ty;
				float w4 = tx  *      ty;
				// 循环处理图像数据
				for (int i = radius; i < src.rows - radius; i++)
				{
					for (int j = radius; j < src.cols - radius; j++)
					{
						// 计算插值
						float t = static_cast<float>(w1*src.at<uchar>(i + fy, j + fx) + w2*src.at<uchar>(i + fy, j + cx) + w3*src.at<uchar>(i + cy, j + fx) + w4*src.at<uchar>(i + cy, j + cx));
						// 进行编码
						dst.at<uchar>(i - radius, j - radius) += ((t > src.at<uchar>(i, j)) || (std::abs(t - src.at<uchar>(i, j)) < std::numeric_limits<float>::epsilon())) << n;
					}
				}
			}
		}


		void getGG() {
			//计算高斯梯度幅值
			magnitude(dx, dy, gg_);
			//std::cout << "gaussian magnitude：" << gg_.rows << "  " << gg_.cols << std::endl;
		}
	};

}