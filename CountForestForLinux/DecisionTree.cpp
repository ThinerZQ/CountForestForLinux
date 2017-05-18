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
		//��ʼ���ڵ㣬�����ڶ���
		Node* currentNode = new Node();
		currentNode->height = currentDepth;


		//����ÿһ���ڵ㶼���ʸ��ΪҶ�ӽڵ�
		//���data�����patch�ı�ǩ��Ƭƫ����һ����ֵ���ж��Ƿ�Ӧ�ó�ΪҶ�ӽڵ�
		
		int majorLabelCount = 0;
		int majorLabelSize = 0;
		
		bool shouldLeaf = currentNode->shouldContinueChooseFeature(data,majorLabelSize,majorLabelCount);

		//����Ҷ�ӽڵ�
		if (sampleSize <= parameter->MinSampleSize || currentDepth >= parameter->MaxDecisionLevels || (currentDepth>parameter->MaxDepthUpper && ! shouldLeaf)) {
	
			currentNode->leaf = true;
			currentNode->height = currentDepth;
			currentNode->InitializeLeaf(data,treeIndex, majorLabelSize, majorLabelCount);
			return currentNode;
		} 
		
		//�ҳ��������
		currentNode->chooseBestFeature_LeftRightNotNull(data);

		cout << "tree, depth, datasize, feature, value: " <<treeIndex<<" , "<< currentNode->height <<" , "<< data.size()<<" , " << currentNode->feature << " , " << currentNode->threshold << endl;

		//������������������ݼ�

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