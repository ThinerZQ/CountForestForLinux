#include <string>
#include <vector>
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <time.h>
#include <utility>
#include <omp.h>
#include <cstdio>
#include <sys/stat.h> 
#include <fstream>
#include <iostream>

#include "TrainingParameters.h"
#include "RandomForest.h"
#include "DensityEstimation.h"
#include "Patch.h"
#include "Utils.h"

#include "Forest.h"



using namespace CrowdCount;
using std::string;
using std::vector;

void GetDirectoryListing(const std::string& path, std::vector<std::string>& filenames);
void train(const string& originImgPath, const string& dottedImgPath, const string& roifilePath, const string& pmapfilePath, const string& modelSavePath);
void printInfo(int i, const string& origImg, time_t start, time_t stop, float num, float gt, Frame* densityMap,vector<int>& predictSequence, vector<float>& predictNums, vector<float>& groundTruthNums, vector<Frame*>& densityMaps);
void test(const string& originImgPath, const string& dottedImgPath, const string& roifilePath, const string& pmapfilePath, const string& loadmodelPath, const string& resultSavePath,const string& dmapSavePath);
int str2int(const char *s);


int main(int argc,char* argv[]){

	omp_set_nested(10);

	//打开输出文件  
	ofstream outf(std::string(argv[2])+"/log.txt");
	//获取cout默认输出  
	streambuf *default_buf = cout.rdbuf();
	//重定向cout输出到文件  
	cout.rdbuf(outf.rdbuf());


	//记录参数
	string datasetName = std::string(argv[2]);

	parameter->NumberOfTrees = str2int(std::string(argv[3]).c_str());
	parameter->NumberOfCandidateFeatures = str2int(std::string(argv[4]).c_str());
	parameter->NumberOfCandidateThresholdsPerFeature = str2int(std::string(argv[5]).c_str());
	parameter->postivePatchNum = str2int(std::string(argv[6]).c_str());
	parameter->patchsize = str2int(std::string(argv[7]).c_str());
	parameter->featureDimension = parameter->patchsize * parameter->patchsize * 5;

	if (datasetName.find("UCSD") != std::string::npos) {
		parameter->height = 158;
		parameter->width = 238;
	}
	else if (datasetName.find("MALL") != std::string::npos) {
		parameter->height = 480;
		parameter->width = 640;
	}
	else {
		cout << "image widht ,height unknown" << endl;
	}


	string trainOriginImgsPath = datasetName + "/train/originImgs";
	string trainDottedImgsPath = datasetName + "/train/dottedImgs";
	string testOriginImgsPath = datasetName + "/test/originImgs";
	string testDottedImgsPath = datasetName + "/test/dottedImgs";
	string roiPath = datasetName.substr(0,datasetName.find_last_of("/")) + "/roi.dat";

	string pmapPath = datasetName.substr(0, datasetName.find_last_of("/")) + "/pmap.dat";
	string modelPath = datasetName + "/rf_lmm.dat";

	string predictMAEResultSavePath = datasetName + "/"+datasetName.substr(datasetName.find_last_of('/')+1,datasetName.size()- datasetName.find_last_of('/'))+"_result_"
		+ std::string(argv[3]).c_str() +"_"
		+ std::string(argv[4]).c_str() +"_"
		+ std::string(argv[5]).c_str() +"_"
		+ std::string(argv[6]).c_str() +".csv";
	string predictDensityMapSavePath = datasetName + "/result/"
		+ std::string(argv[3]).c_str() + "_"
		+ std::string(argv[4]).c_str() + "_"
		+ std::string(argv[5]).c_str() + "_"
		+ std::string(argv[6]).c_str() + "/";


	

	if (std::string(argv[1]) == "train") {
		cout << "Train ,DataSet = " << datasetName << endl;
		parameter->print();

		train(trainOriginImgsPath, trainDottedImgsPath,roiPath,pmapPath,modelPath);
	}
	else{
		cout << "Predict ,DataSet = " << datasetName << endl;
		parameter->print();
		//该文件夹不存在，创建一个
		if (access(predictDensityMapSavePath.c_str(), 0) == -1) {
			int flag = mkdir(predictDensityMapSavePath.c_str(), 0777);
			if (flag != 0) {
				cout << "create predict density map dir faild : " << predictDensityMapSavePath << endl;
			}
		}

		test(testOriginImgsPath, testDottedImgsPath, roiPath, pmapPath, modelPath, predictMAEResultSavePath, predictDensityMapSavePath);
	}
	cout.rdbuf(default_buf);
	return 0;
}

void train(const string& originImgPath, const string& dottedImgPath, const string& roifilePath,const string& pmapfilePath, const string& modelSavePath){
	time_t start, stop;
	start = time(NULL);
	vector<string> origImgs;
	GetDirectoryListing(originImgPath, origImgs);
	vector<string> dottedImgs;
	GetDirectoryListing(dottedImgPath, dottedImgs);

	std::vector<Patch*> trainingData;

	Utils::load(origImgs, dottedImgs, roifilePath, pmapfilePath, trainingData);

	

	Forest* forest = DensityEstimation::TrainForest(trainingData);
	forest->Serialize(modelSavePath);
	stop = time(NULL);
	cout<<"Use Time : "<<(stop - start)<<endl;


}


void test(const string& originImgPath, const string& dottedImgPath, const string& roifilePath, const string& pmapfilePath, const string& loadmodelPath, const string& resultSavePath, const string& dmapSavePath){


	//生成测试的的DataPointCollection
	time_t entire_start, entire_stop;
	entire_start = time(NULL);
	vector<string> origImgs;
	GetDirectoryListing(originImgPath, origImgs);
	vector<string> dottedImgs;
	GetDirectoryListing(dottedImgPath, dottedImgs);


	Forest* forest=Forest::Deserialize(loadmodelPath);


	cv::Mat roimask = Utils::loadROI(roifilePath);
	cv::Mat pmap = Utils::loadPmap(pmapfilePath);



	vector<int> predictSequence;
	vector<float> predictNums;
	vector<float> groundTruthNums;
	vector<Frame*> densityMaps;

#pragma omp parallel for ordered num_threads(10)
	for (int w = 0; w < origImgs.size(); w++){
		time_t start, stop;
		start = time(NULL);

		std::vector<Patch*> testData;

		Utils::load(origImgs[w], dottedImgs[w],roimask ,pmap, testData);
		
	//	cout << "load done" << endl;
		RandomForest::predict(forest, testData);
		//cout << "predict done" << endl;
		int ih = parameter->height;
		int iw = parameter->width;
		int dx = parameter->patchsize / 2;
		int dy = parameter->patchsize / 2;
		int stride = parameter->stride;
		cv::Mat densitymap = cv::Mat::zeros(ih, iw, CV_32F);
		cv::Mat counter = cv::Mat::zeros(ih, iw, CV_32F);
		std::vector<std::pair<int, int>>  patchcenters;
		//然后将张图片所有的块的密度图融合成整张图的密度图
		for (int x = dx; x < ih - dx; x = x + stride)
		{
			for (int y = dy; y < iw - dy; y = y + stride)
			{
				patchcenters.push_back(std::make_pair(x, y));
			}
		}
		
		for (int k = 0; k < patchcenters.size(); k++)
		{
			int x = patchcenters[k].first;
			int y = patchcenters[k].second;
			cv::Mat patchDensity;
			cv::GaussianBlur(testData[k]->predictLabel, patchDensity, cv::Size(9, 9), 0.2*pmap.at<float>(x,y));
			for(int i = x - dx; i < x + dx + 1; i++){
				for (int j = y - dy; j < y + dy + 1; j++){
					densitymap.at<float>(i, j) = densitymap.at<float>(i, j) + patchDensity.at<float>(i - x + dx, j - y + dy);
					counter.at<float>(i,j) = counter.at<float>(i,j) + 1;
				}
			}
			
		}
	
		for (int r = 0; r < densitymap.rows; r++)
		{
			for (int c = 0; c < densitymap.cols; c++)
			{
				densitymap.at<float>(r, c) = densitymap.at<float>(r, c) / counter.at<float>(r, c);
			}
		}




		float num = 0;
		for (int r = 0; r < densitymap.rows; r++)
		{
			for (int c = 0; c < densitymap.cols; c++)
			{
				if (unsigned(roimask.at<uchar>(r, c)) == 1)
					num += densitymap.at<float>(r, c);
				else
					densitymap.at<float>(r, c) = 0;
			}
		}
		GaussianBlur(densitymap, densitymap, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);
		string filename = dottedImgs[w].substr(dottedImgs[w].find_last_of('/') + 1, dottedImgs[w].size() - dottedImgs[w].find_last_of('/'));
		string savefilename = filename.replace(filename.find(".png"),4,".dat");
		Frame* densityMapFrame = new Frame(savefilename,densitymap);

		//std::cout << savefilename << std::endl;

		cv::Mat dottedImg = cv::imread(dottedImgs[w], 0);
		int gt = 0;
		for (int r = 0; r < dottedImg.rows; r++)
		{
			for (int c = 0; c < dottedImg.cols; c++)
			{
				dottedImg.at<uchar>(r, c) = dottedImg.at<uchar>(r, c) / 72;
				if (unsigned(dottedImg.at<uchar>(r, c)) == 1 && unsigned(roimask.at<uchar>(r,c)) == 1)
					gt++;
				 
			}
		}

		//释放testData和其中的指针
		for (vector<Patch*>::iterator it = testData.begin(); it != testData.end(); it++)
			if (NULL != *it)
			{
				delete *it;
				*it = NULL;
			}
		testData.clear();

		stop = time(NULL);
		 
		//从这里开始就要顺序加入了，使用order子句。
#pragma omp critical
		printInfo(w, origImgs[w], start, stop, num, gt, densityMapFrame, predictSequence, predictNums, groundTruthNums,densityMaps);
		
	}
	Utils::saveResult(resultSavePath, dmapSavePath,predictSequence,predictNums, groundTruthNums, densityMaps);
	entire_stop = time(NULL);
	cout << "Use Time : " << (entire_stop - entire_start) << endl;
}
void printInfo(int i,const string& origImg, time_t start, time_t stop,float num,float gt, Frame* densityMap, vector<int>& predictSequence, vector<float>& predictNums, vector<float>& groundTruthNums, vector<Frame*>& densityMaps) {
	//std::cout << "loading and preprocess frame" << origImg << std::endl;
	//std::cout << "Estimate " << i << std::endl;
	std::cout << "frame"<< i<<" , ( " << gt << " , " << num << " , "<<(gt-num)<<" ) ,time : " << (stop - start) << std::endl;
	predictSequence.push_back(i);
	predictNums.push_back(num);
	groundTruthNums.push_back(gt);
	densityMaps.push_back(densityMap);
}

void GetDirectoryListing(const std::string& path, std::vector<std::string>& filenames)
{
	DIR *dp;
	struct dirent *dirp;
	if ((dp = opendir(path.c_str())) == NULL)
		throw std::runtime_error("Failed to obtain directory listing.");

	try
	{
		while ((dirp = readdir(dp)) != NULL)
		{
			std::string name = dirp->d_name;
			if (name == "." || name == "..")
				continue;
		/*	if (extension != "" && name.substr(name.size() - 4, 4) != extension)
				continue;*/
			filenames.push_back(path+'/'+name);
		}
	}
	catch (...)
	{
		closedir(dp);
		throw;
	}

	closedir(dp);
	sort(filenames.begin(), filenames.end());
	
}
int str2int(const char *s) {
	int sign = 1, value = 0;
	if (*s == '+') {
		++s;
	}
	else if (*s == '-') {
		++s;
		sign = -1;
	}
	while (*s) {
		if (*s >= '0' && *s <= '9') {
			value = value * 10 + *s - '0';
			++s;
		}
		else {
			break;
		}
	}
	return sign * value;
}
