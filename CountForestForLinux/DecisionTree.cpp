#include "DecisionTree.h"
#include "TrainingParameters.h"
namespace CrowdCount{

	Tree* DecisionTree::train(const std::vector<Patch*>& data, int& treeIndex)
	{
		Tree* tree =new Tree();
		std::cout << "start train th " << treeIndex << " tree" << endl;
		Node* root = trainRecursive(data, 0,treeIndex);
		tree->root = root;
		return tree;
	}
	Node* DecisionTree::trainRecursive(const std::vector<Patch*>& data, int currentDepth,int& treeIndex) {

		long sampleSize = data.size();
		//初始化节点，分配在堆上
		Node* currentNode = new Node();
		currentNode->height = currentDepth;


		//不是每一个节点都有资格成为叶子节点
		//如果data里面的patch的标签普片偏向于一个数值，判断是否应该成为叶子节点
		
		int majorLabelCount = 0;
		int majorLabelSize = 0;
		
		bool shouldLeaf = currentNode->shouldContinueChooseFeature(data,majorLabelSize,majorLabelCount);

		//处理叶子节点
		if (sampleSize <= parameter->MinSampleSize || currentDepth >= parameter->MaxDecisionLevels || (currentDepth>parameter->MaxDepthUpper && ! shouldLeaf)) {
	
			currentNode->leaf = true;
			currentNode->height = currentDepth;
			currentNode->InitializeLeaf(data,treeIndex, majorLabelSize, majorLabelCount);
			return currentNode;
		} 
		
		//找出最佳特征
		currentNode->chooseBestFeature_LeftRightNotNull(data);

		cout << "tree, depth, datasize, feature, value: " <<treeIndex<<" , "<< currentNode->height <<" , "<< data.size()<<" , " << currentNode->feature << " , " << currentNode->threshold << endl;

		//根据最佳特征划分数据集

		std::vector<Patch*> leftData;
		std::vector<Patch*> rightData;

		currentNode->splitDataSet(data, leftData, rightData);
	
		currentNode->left = trainRecursive(leftData, currentDepth + 1,treeIndex);
		currentNode->right = trainRecursive( rightData, currentDepth + 1,treeIndex);
		return currentNode;
	}

	 std::vector<cv::Mat> DecisionTree::predict(Tree* tree, std::vector<Patch*>& testData)
	{
		std::vector<cv::Mat> predictLabel = std::vector<cv::Mat>();
		for (size_t i = 0; i < testData.size(); i++)
		{
			predictLabel.push_back(predictWithOnePatch(tree->root, testData[i]));
		}
		return predictLabel;
	}
	cv::Mat DecisionTree::predictWithOnePatch(Node* node, Patch* patch) {
		int currentFeature = node->feature;
		double bestSplitVal = node->threshold;

		double currentFeatureVal = patch->feature[currentFeature];

		if (node->left == NULL && node->right == NULL) {
			return node->density;
		}
		if (currentFeatureVal <= bestSplitVal) {
			return predictWithOnePatch(node->left, patch);
		}
		else {
			return predictWithOnePatch(node->right, patch);
		}
	}


}