#include "Patch.h"
#include "Node.h"
#include "Utils.h"
#include <omp.h>
#include "TrainingParameters.h"
#include <iostream>

namespace CrowdCount {
	class Patch;
	void Node::InitializeLeaf(const std::vector<Patch *> & data,int treeIndex,int majorlabelsize,int majorlabelcount)
	{
		//这里形成新的叶子节点，并将其核密度图存入
		
		//找出所有majorlabelsize 个人的Patch
		std::vector<Patch*> majorPatches;
		for (int index = 0; index < data.size(); index++)
		{
			if (data[index]->peopleCount == majorlabelsize) {
				majorPatches.push_back(data[index]);
			}
		}

		cv::Mat tempDensity = cv::Mat::zeros(parameter->patchsize, parameter->patchsize, CV_32F);
		if (majorlabelsize != 0)
		{
			//这里做k-means后再用高斯模糊
			cv::Mat points(majorlabelcount * majorlabelsize, 1, CV_32FC2);
			cv::Mat centers(majorlabelsize, 1, points.type()), labels;    //用来存储聚类后的中心点 
			int m = 0;
			for (int i = 0; i < majorPatches.size(); i++)
			{
				for (int j = 0; j < majorPatches[i]->label.rows; j++)
				{
					for (int k = 0; k < majorPatches[i]->label.cols; k++)
					{
						if (unsigned(majorPatches[i]->label.at<uchar>(j, k)) == 1) {
							points.at<cv::Vec2f>(m++, 0) = cv::Vec2f(j, k);

						}
					}
				}
			}
			kmeans(points, majorlabelsize, labels,
				cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
				3, cv::KMEANS_PP_CENTERS, centers);
			
			cout << "tree, depth, datasize, majorsz, majct: " <<treeIndex<<" , "<< height <<" , "<< data.size() << " , " << majorlabelsize << "-------------------------------------------------->" << majorlabelcount <<" , "<<(majorlabelcount*1.0/(data.size()==0?1:data.size())*majorlabelcount)<< std::endl;
			for (int i = 0; i < majorlabelsize; i++)
			{
				//std::cout << "(" << centers.at<cv::Vec2f>(i, 0)[0] << "," << centers.at<cv::Vec2f>(i, 0)[1] << ")" << std::endl;
				tempDensity.at<float>(int(centers.at<cv::Vec2f>(i, 0)[0]), int(centers.at<cv::Vec2f>(i, 0)[1])) = 1;
			}
			//cv::GaussianBlur(tempDensity, tempDensity, cv::Size(3, 3), 0.1);
			//cout << "leaf density (default) = " << endl << tempDensity << endl << endl;
		}
		this->density = tempDensity;
	}

	void Node::chooseBestFeature_LeftRightNotNull( const std::vector<Patch*>& data) {


		chooseBestFeature(data);

		std::vector<Patch*> left;
		std::vector<Patch*> right;
		splitDataSet(data, left, right);


		
		
		if (this-> height > parameter->MaxDepthUpper) {
			//下层

			int leftMajorLabelSize, rightMajorLabelSize;
			int leftMajorLabelCount, rightMajorLabelCount;

			bool leftContinue = shouldContinueChooseFeature(left, leftMajorLabelSize, leftMajorLabelCount);
			bool rightContinue = shouldContinueChooseFeature(right, rightMajorLabelSize, rightMajorLabelCount);

			int loop = 0;
			while ((leftMajorLabelSize == rightMajorLabelSize) &&  loop <= 5)
			{
				loop++;
				left.clear();
				right.clear();
				chooseBestFeature(data);
				splitDataSet(data, left, right);
				leftContinue = shouldContinueChooseFeature(left, leftMajorLabelSize, leftMajorLabelCount);
				rightContinue  = shouldContinueChooseFeature(right, rightMajorLabelSize, rightMajorLabelCount);
			}
			if (loop >= 5) {
				std::cout << "depper level choose best Feature more 5 times, chooseBestFeature again" << std::endl;
				chooseBestFeature(data);
			}
		}
		else {
			//上层
			int loop = 0;
			while ((left.size() == 0 || right.size() == 0) && loop <= 5)
			{
				loop++;
				left.clear();
				right.clear();
				chooseBestFeature(data);
				splitDataSet(data, left, right);
			}
			if (loop >= 5) {
				std::cout << "upper level choose best Feature more 5 times, chooseBestFeature again" << std::endl;
				chooseBestFeature(data);
			}
		}
	}

	bool Node::shouldContinueChooseFeature(const std::vector<Patch*>& data,int& majorLabelSize,int& majorLabelCount) {

		if (data.size() == 0) {
			majorLabelSize = 0;
			majorLabelCount = 0;
			return true;
		}

		std::vector<int> labelsizelist;
		for (int index = 0; index < data.size(); index++)
		{
			labelsizelist.push_back(data[index]->peopleCount);
		}

		std::map<int, int> labelcount;
		for (int i = 0; i < labelsizelist.size(); i++)
		{
			labelcount[labelsizelist[i]]++;
		}
		
		for (auto iter = labelcount.begin(); iter != labelcount.end(); iter++)
		{
			if (iter->second > majorLabelCount)
			{
				majorLabelCount = iter->second;
				majorLabelSize = iter->first;
			}
		}
		bool shouldContinue = true;
		
		if ((majorLabelCount / data.size()) >=0.8) {
			shouldContinue = false;
		}
		return shouldContinue;
	}


	//选择出当前节点的最佳特征
	void Node::chooseBestFeature(const std::vector<Patch*>& data) {
		
		double finalBestGrain = 100000000000000;
		int finalBestFeature = 0;
		double finalBestSpiltVal = 0;
		srand((unsigned)time(0));
		// 根据训练参数中的MaxFeature找出最佳特征
#pragma omp parallel for num_threads(parameter->NumberOfCandidateFeatures)
		for (int i = 0; i < parameter->NumberOfCandidateFeatures; i++)
		{

			double bestGrain4Feature = 100000000000000;
			int bestFeature;
			double bestSpiltVal = 0.0;

			
			int randFeature = rand() % parameter->featureDimension;
			cv::Mat sumHistorgram = sumHistogram(data);
			//随机取一个特征
			for (size_t j = 0; j < parameter->NumberOfCandidateThresholdsPerFeature; j++)
			{
				//取随机当前特征的一个特征值
				
				int index = rand() % data.size();
				double randFeatureValue = data[index]->feature[randFeature];
				double tempGrain = 0;

				std::vector<Patch*> left;
				std::vector<Patch*> right;
				splitDataSet(data, randFeature, randFeatureValue, left, right);

				if (height <= parameter->MaxDepthUpper) {
					tempGrain = informationGrain(left, right, sumHistorgram);
				}
				else {
					tempGrain = informationGrain2(data, left, right);
					//cout << "tempGrain2 = " << tempGrain<<endl;
				}

				if (tempGrain <= bestGrain4Feature ) {
					bestGrain4Feature = tempGrain;
					bestFeature = randFeature;
					bestSpiltVal = randFeatureValue;
				}
			
			}
			sumHistorgram.release();
			//cout << bestGrain4Feature <<", " << bestFeature << "," << bestSpiltVal << endl;
			//原子形式执行
			#pragma omp critical
			if (bestGrain4Feature <= finalBestGrain ) {
				
					finalBestFeature = bestFeature;
					finalBestSpiltVal = bestSpiltVal;
			}

		}

		this->feature = finalBestFeature;
		this->threshold = finalBestSpiltVal;

	}
	void Node::splitDataSet(const std::vector<Patch* >& patches, int randFeature, double randFeatureValue, std::vector<Patch* >& left, std::vector<Patch* >& right) {
		for (int i = 0; i < patches.size(); i++) {
			if (patches[i]->feature[randFeature] <= randFeatureValue) {
				left.push_back(patches[i]);
			}
			else {
				right.push_back(patches[i]);
			}
		}
	}
	void Node::splitDataSet(const std::vector<Patch*>& patches, std::vector<Patch*>& left, std::vector<Patch*>& right) {

		for (int i = 0; i < patches.size(); i++) {
			if (patches[i]->feature[this->feature] <= this->threshold) {
				left.push_back(patches[i]);
			}
			else {
				right.push_back(patches[i]);
			}
		}
	}
	

	double Node::informationGrain(std::vector<Patch*>& left, std::vector<Patch*>& right, cv::Mat& sumHistorgram) {

		//求size 比较小的一边的总直方图，然后用sumHistogram减去它得到另一个
		cv::Mat leftHistorgramAvg;
		cv::Mat rightHistorgramAvg;
		if (left.size() < right.size()) {
			leftHistorgramAvg = sumHistogram(left);
			cv::subtract(sumHistorgram, leftHistorgramAvg, rightHistorgramAvg);
		}
		else {
			rightHistorgramAvg = sumHistogram(right);
			cv::subtract(sumHistorgram, rightHistorgramAvg, leftHistorgramAvg);
		}
		cv::divide(left.size(), leftHistorgramAvg, leftHistorgramAvg);
		cv::divide(right.size(), rightHistorgramAvg, rightHistorgramAvg);

		double leftFrobenius = 0;
		double rightFrobenius = 0;
		for (int i = 0; i < left.size(); i++) {
			cv::Mat tempHistogram = left[i]->histogram;
			cv::subtract(tempHistogram, leftHistorgramAvg, tempHistogram);
			leftFrobenius += cv::norm(tempHistogram, 4);
		}
		
		for (int i = 0; i < right.size(); i++) {
			cv::Mat tempHistogram = right[i]->histogram;
			cv::subtract(tempHistogram, rightHistorgramAvg, tempHistogram);
			rightFrobenius += cv::norm(tempHistogram, 4);
		}

		cv::subtract(leftHistorgramAvg, rightHistorgramAvg, leftHistorgramAvg);
		double temp = cv::norm(leftHistorgramAvg, 4);

		leftHistorgramAvg.release();
		rightHistorgramAvg.release();
		
		return rightFrobenius + leftFrobenius + temp;
	}
	cv::Mat Node::avgHistogram(std::vector<Patch*>& patches) {

		//TODO 堆上创建
		cv::Mat avgHistogram= cv::Mat::zeros(32, 1, CV_8U);;
		for (int i = 0; i < patches.size(); i++) {
			cv::add(avgHistogram, patches[i]->histogram, avgHistogram);
		}
		cv::divide(patches.size(), avgHistogram, avgHistogram);
		//TODO: 内存问题
		return avgHistogram;

	}
	cv::Mat Node::sumHistogram(const std::vector<Patch*>& patches) {

		//TODO 堆上创建
		cv::Mat sumHistogram = cv::Mat::zeros(32, 1, CV_8U);;
		for (int i = 0; i < patches.size(); i++) {
			cv::add(sumHistogram, patches[i]->histogram, sumHistogram);
		}
		//TODO: 内存问题
		return sumHistogram;

	}

	double Node::informationGrain2(const std::vector<Patch*> & patches, std::vector<Patch*> & left, std::vector<Patch*> & right) {
		std::vector<int> leftLabelSizeList;
		std::vector<int> rightLabelsSizeList;
		getLabelSizeList(left,leftLabelSizeList);
		getLabelSizeList(right,rightLabelsSizeList);

		double leftgrain = calcShannonEnt(leftLabelSizeList)*left.size() / patches.size();
		double rightgrin = calcShannonEnt(rightLabelsSizeList)*right.size() / patches.size();

		return leftgrain + rightgrin;

	}
	void Node::getLabelSizeList(const std::vector<Patch*>& patches, std::vector<int>& labelSizeList) {
		for (int i = 0; i < patches.size(); i++) {
			//if (countNonZero(patches[i]->label) !=0) {
				labelSizeList.push_back(patches[i]->peopleCount);
			//}
		}
	}
	double Node::calcShannonEnt(const std::vector<int>& labelSizeList) {
		int size = labelSizeList.size();

		std::map<int, int> count;
		for (int i = 0; i < labelSizeList.size(); i++)
		{
			count[labelSizeList[i]]++;
		}

		double shannonEnt = 0;
		double prob = 0.0;

		for (auto iter = count.begin(); iter != count.end(); iter++)
		{
			prob = (double)iter->second / size;
			shannonEnt -= prob * (log(prob) / log((double)2));
			return shannonEnt;
		}
	}


		void Node::Serialize(std::ostream& o) const
		{
			o.write((const char*)&leaf, sizeof(bool));
			o.write((const char*)&height, sizeof(height));
			o.write((const char*)&feature, sizeof(int));
			o.write((const char*)&threshold, sizeof(threshold));
			// 叶子节点
			if (leaf) {
				
				//unsigned datalen = density.dataend - density.datastart;
				int cols = density.cols;
				int rows = density.rows;
				bool continuous = density.isContinuous();
				int data_size = cols * rows * density.elemSize();
				o.write((const char*)&data_size, sizeof(data_size));
				//cout << "leaf density (default) = " << endl << density << endl << endl;
				if (continuous)
				{
					o.write((const char*)density.datastart, data_size);
				}
				else {
					const unsigned int row_size = cols*density.elemSize();
					for (int i = 0; i < rows; i++) {
						o.write((const char*)density.ptr(i), row_size);
					}
				}
				
			}
			else {
				//非叶子节点
				left->Serialize(o);
				right->Serialize(o);
			}
		}

		Node* Node::Deserialize(std::istream& i)
		{
			Node* node = new Node();
			
			i.read((char*)&node->leaf, sizeof(bool));
			i.read((char*)&node->height, sizeof(height));
			i.read((char*)&node->feature, sizeof(int));
			i.read((char*)&node->threshold, sizeof(threshold));

			if (node->leaf)
			{
			
				int datalen = 0;
				
				i.read((char*)&datalen, sizeof(datalen));
				char*  data = new char[datalen];
				i.read(data, datalen);
				
				int patchSize = sqrt(double(datalen / 4));
				node->density = cv::Mat(patchSize, patchSize, CV_32F);

				for (int k = 0; k < patchSize; k++)
				{
					for (int j = 0; j < patchSize; j++)
					{
						float temp = *(float*)(data);
						node->density.at<float>(k, j) = temp;
						//cout << temp << ",";
						data += 4;
					}
					//cout << endl;
				}
				//cout << endl;
				//cout << "leaf density (default) = " << endl << node->density << endl << endl;
			}
			else {
				
				node->left = Node::Deserialize(i);
				node->right = Node::Deserialize(i);
			}
			return node;
		}

}