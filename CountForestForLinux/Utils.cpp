#include <opencv2/opencv.hpp>
#include <unistd.h>  
#include <sys/stat.h>  
#include <sys/types.h>
#include <fstream>
#include <iostream>


#include "TrainingParameters.h"

#include "Utils.h"


namespace CrowdCount {
	using std::cout;
	using std::endl;
	void Utils::load(const std::vector<std::string>& originimgs, const std::vector<std::string>&  dottedimgs, const std::string&  roipath, const std::string&  pmappath, std::vector<Patch*>& patches) {
		//将每一张图片的相关信息放在Frame对象中,对特征图和标记图进行划分后存放在DataCollection的feature和label中
		srand((unsigned)time(0));
		//接下来做划分和存储
		int dx = parameter->patchsize / 2;
		int dy = parameter->patchsize / 2;

		cv::Mat roimask = loadROI(roipath);

		cv::Mat pmap = loadPmap(pmappath);
	
	  

#pragma omp parallel for num_threads(10)
		for (int i = 0; i < originimgs.size(); i++)
		{
			cout << "loading frame " << i << endl;
			Frame frame(originimgs[i], pmap,roimask);
			cv::Mat dottedimg = cv::imread(dottedimgs[i], 0);
			for (int r = 0; r < dottedimg.rows; r++)
			{
				for (int c = 0; c < dottedimg.cols; c++)
				{
					dottedimg.at<uchar>(r, c) = dottedimg.at<uchar>(r, c) / 72;
				}

			}
			//随机采样
			//cout << "patchNum = " << parameters.patchNum;
			for (int patchindex = 0; patchindex < parameter->patchNum; patchindex++)
			{
				//随机得到Patch的中心坐标
				unsigned int y = dy + rand()%(frame.cols_ - dy*2);
				unsigned int x = dx + rand() % (frame.rows_ - dx * 2);

				////根据中心坐标获取其label
				cv::Mat patchlabel = dottedimg(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));

				////根据中心坐标获取其roi区域
				cv::Mat patchROI = roimask(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
				//cv::Mat patchOrig = frame.originImg_(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));

				////如果其roi区域内没有人，就继续再随机获取其中心坐标
				//if (patchindex % 2 == 0 && patchindex < parameter->postivePatchNum*2) {
				//	while (countNonZero(patchOrig) == 0 || countNonZero(patchlabel) == 0) {

				//		//随机得到Patch的中心坐标
				//		y = dy + rand() % (frame.cols_ - dy * 2);
				//		x = dx + rand() % (frame.rows_ - dx * 2);

				//		//根据中心坐标获取其label
				//		patchlabel = dottedimg(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));

				//		patchOrig = frame.originImg_(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
				//	}
				//}



				if (patchindex < parameter->postivePatchNum) {
				/*	int peoplecount = 0;
					if (patchindex >= 0 && patchindex < (4 / 10)* parameter->postivePatchNum) {
						//取只有1个人的
						peoplecount = 1;
					}
					else if (patchindex >= (4 / 10)* parameter->postivePatchNum && patchindex < (7 / 10)* parameter->postivePatchNum) {
						//去只有2个人的
						peoplecount = 2;
					}
					else if (patchindex >= (7 / 10)* parameter->postivePatchNum && patchindex < (9 / 10)* parameter->postivePatchNum) {
						//去只有3个人的
						peoplecount = 3;
					}
					else{
						//(patchindex >= (9 / 10)* parameter->postivePatchNum && patchindex <=parameter->postivePatchNum)
						//去只有4个人
						peoplecount = 4;
					}
					int loop;*/
					//取roi里面有人的
					while (countNonZero(patchlabel) == 0) {

						//loop++;
						y = dy + rand() % (frame.cols_ - dy * 2);
						x = dx + rand() % (frame.rows_ - dx * 2);

						patchlabel = dottedimg(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
						patchROI = roimask(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
					}
				}
				else {
					//取roi里面没人的
					while (countNonZero(patchROI)==0 && countNonZero(patchlabel) != 0) {
						y = dy + rand() % (frame.cols_ - dy * 2);
						x = dx + rand() % (frame.rows_ - dx * 2);
						patchlabel = dottedimg(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
						patchROI = roimask(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
					}
				}



				

				//分配在堆上
				Patch* patch = new Patch(x, y);
				patch->label = patchlabel;
				patch->calHistogram(pmap.at<float>(x, y));

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.log_.ptr<float>(r) + y - dy, frame.log_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.lbp_.ptr<float>(r) + y - dy, frame.lbp_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.gg_.ptr<float>(r) + y - dy, frame.gg_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.structVal_.ptr<float>(r) + y - dy, frame.structVal_.ptr<float>(r) + y + 2 * parameter->patchsize - dy);
				/*std::cout << features.size() << std::endl;*/
#pragma omp critical
				patches.push_back(patch);

			}

		}

	}
	void Utils::load(const std::string& originimg, const std::string& dottedimg,cv::Mat& roi,cv::Mat&  pmap, std::vector<Patch*>& patches) {


		//将每一张图片的相关信息放在Frame对象中,对特征图和标记图进行划分后存放在DataCollection的feature和label中
		

		Frame frame(originimg, pmap,roi);
		
		
		cv::Mat dottedImg = cv::imread(dottedimg, 0);

		for (int r = 0; r < dottedImg.rows; r++)
		{
			for (int c = 0; c < dottedImg.cols; c++)
			{
				dottedImg.at<uchar>(r, c) = dottedImg.at<uchar>(r, c) / 72;
			}

		}
		
		//密集采样
		int stride = parameter->stride;
		int ih = frame.rows_;
		int iw = frame.cols_;
		int dx = parameter->patchsize / 2;
		int dy = parameter->patchsize / 2;

		for (int x = dx; x < ih - dx; x = x + stride)
		{
			for (int y = dy; y < iw - dy; y = y + stride)
			{
				Patch* patch = new Patch();
				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.log_.ptr<float>(r) + y - dy, frame.log_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.lbp_.ptr<float>(r) + y - dy, frame.lbp_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.gg_.ptr<float>(r) + y - dy, frame.gg_.ptr<float>(r) + y + dy + 1);
				/*std::cout << features.size() << std::endl;*/

				for (int r = x - dx; r < x + dx + 1; r++)
					patch->feature.insert(patch->feature.end(), frame.structVal_.ptr<float>(r) + y - dy, frame.structVal_.ptr<float>(r) + y + 2 * parameter->patchsize - dy);
				/*std::cout << features.size() << std::endl;*/
				cv::Mat patchlabel = dottedImg(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1));
				patch->label = patchlabel;
				/*std::cout << labels.size() << std::endl;*/
				patch->hasForeground = countNonZero(frame.originImg_(cv::Range(x - dx, x + dx + 1), cv::Range(y - dy, y + dy + 1)));
				patches.push_back(patch);
			}
		}
	}
	cv::Mat Utils::loadROI(const std::string& roipath) {

		std::ifstream  file(roipath);
		int temp;
		cv::Mat roimask = cv::Mat::zeros(parameter->height, parameter->width, CV_8U);
		for (int r = 0; r < parameter->height; r++)
		{
			for (int c = 0; c < parameter->width; c++)
			{
				file >> temp;
				roimask.at<uchar>(r, c) = uchar(temp);
			}
		}
		file.close();
		return roimask;

	}
	cv::Mat Utils::loadPmap(const std::string& pmappath) {

		std::ifstream  file(pmappath);
		float temp;
		cv::Mat pmap = cv::Mat::zeros(parameter->height, parameter->width, CV_32F);
		for (int r = 0; r < parameter->height; r++)
		{
			for (int c = 0; c < parameter->width; c++)
			{
				file >> temp;
				pmap.at<float>(r, c) = temp;
			}
		}
	
		file.close();
		return pmap;
	}
	void Utils::saveResult(const std::string& path, const string& dmapSavePath,vector<int>& predictSequence, vector<float>& predictNums, vector<float>& groundTruthNums, vector<Frame*>& densityMaps) {


		for (int i = 0; i < predictSequence.size() - 1; i++) {
			for (int j = 0; j < predictSequence.size() - i - 1; j++) {
				if (predictSequence[j] > predictSequence[j + 1]) {
					// 交换predictSequence
					int temp = predictSequence[j];
					predictSequence[j] = predictSequence[j + 1];
					predictSequence[j + 1] = temp;

					// 交换groundTruthNums
					float temp1 = groundTruthNums[j];
					groundTruthNums[j] = groundTruthNums[j + 1];
					groundTruthNums[j + 1] = temp1;

					// 交换predictNums
					temp1 = predictNums[j];
					predictNums[j] = predictNums[j + 1];
					predictNums[j + 1] = temp1;
				}
			}
		}
		float sum = 0;
		//计算mae
		for (size_t i = 0; i < predictNums.size(); i++)
		{
			sum += fabs(predictNums[i] - groundTruthNums[i]);
		}
		float mae = sum / predictNums.size();

		//更新结果的保存路径，里面加上mae;
		string tempPath = path;
		string rep = "result_" + to_string(mae);
		tempPath.replace(path.find("result"), 6, rep);
		std::ofstream o(tempPath.c_str(), std::ios_base::trunc);
		for (size_t i = 0; i < predictNums.size(); i++)
		{
			o << groundTruthNums[i] << "," << predictNums[i] << "\n";
		}
		o << "\n\n";
		o << "mae," << mae;
		o.close();//关闭文件

		cout << "save predict result to " << tempPath << endl;


		time_t start, stop;
		start = time(NULL);

		string realDensityMapSavePath = dmapSavePath + "MAE_"+to_string(mae) + "/";
		//该文件夹不存在，创建一个
		if (access(realDensityMapSavePath.c_str(), 0) == -1) {
			int flag = mkdir(realDensityMapSavePath.c_str(), 0777);
			if (flag != 0) {
				cout << "create predict density map dir faild : " << realDensityMapSavePath<<endl;
			}
		}
		
		//并发保存保存密度图
#pragma omp parallel for
		for (int k = 0; k < densityMaps.size(); k++) {
			//cout << densityMaps[k]->densityMap;
			std::ofstream fout;
			fout.open(realDensityMapSavePath+ densityMaps[k]->densityMapPath);
			fout << densityMaps[k]->densityMap.rows << std::endl;
			fout << densityMaps[k]->densityMap.cols << std::endl;
			for (int i = 0; i < densityMaps[k]->densityMap.rows; i++) {
				for (int j = 0; j < densityMaps[k]->densityMap.cols; j++) {

					fout << densityMaps[k]->densityMap.at<float>(i, j) << "\t";
				}
				fout << std::endl;
			}
			
			//fout << std::flush;
			fout.close();
		}
		/*for (vector<Frame*>::iterator it = densityMaps.begin(); it != densityMaps.end(); it++) {
			if (NULL != *it)
			{
				delete *it;
				*it = NULL;
			}
		}
		densityMaps.clear();*/
		stop = time(NULL);
		cout << "save predict density map to : " << realDensityMapSavePath << endl;
		cout << " save predict density map time use: " << (stop - start) << endl;
	}
	
};
