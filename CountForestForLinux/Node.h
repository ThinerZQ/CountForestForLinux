#pragma once
#include <opencv2/opencv.hpp>
#include <map>
#include <vector>

namespace CrowdCount{
	class Patch;
	class Node 
	{
	public:
		int feature;
		float threshold;
		int height;
		bool leaf;
		Node* left;
		Node* right;
		cv::Mat density;

		Node()
		{
			left = NULL;
			right = NULL;
			leaf = false;
			feature = 0;
			threshold = 0;
			height = 0;
		}
		Node(int feature, float threshold)
		{
			feature = feature;
			threshold = threshold;
		}

		void InitializeLeaf(const std::vector<Patch *> & data, int treeIndex, int majorlabelsize, int majorlabelcount);

		void chooseBestFeature_LeftRightNotNull( const std::vector<Patch*>& data);
		void chooseBestFeature(const std::vector<Patch*>& data);
		bool shouldContinueChooseFeature(const std::vector<Patch*>& data, int& majorLabelSize, int& majorLabelCount);

		void splitDataSet(const std::vector<Patch* >& patches, int randFeature, double randFeatureValue, std::vector<Patch* >& left, std::vector<Patch* >& right);
		void splitDataSet(const std::vector<Patch*>& patches, std::vector<Patch*>& left, std::vector<Patch*>& right);
		void Serialize(std::ostream& o) const;
		static Node* Deserialize(std::istream& i);

		private :
			double informationGrain(std::vector<Patch*>& left, std::vector<Patch*>& right, cv::Mat& sumHistorgram);
			cv::Mat avgHistogram(std::vector<Patch*>& patches);
			cv::Mat sumHistogram(const std::vector<Patch*>& patches);
			double informationGrain2(const std::vector<Patch*> & patches, std::vector<Patch*> & left, std::vector<Patch*> & right);
			void getLabelSizeList(const std::vector<Patch*>& patches, std::vector<int>& labelSizeList);
			double calcShannonEnt(const std::vector<int>& labelSizeList);	
	};
}
