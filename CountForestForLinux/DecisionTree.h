#pragma once
#include "Forest.h"
#include "Tree.h"
#include "Node.h"
#include "Patch.h"


namespace CrowdCount {

	class Patch;
	class Node;
	class Tree;
	class DecisionTree
	{
	public:
		
		static Tree* train(const std::vector<Patch*>& data,int& treeIndex);
		static Node* trainRecursive( const std::vector<Patch*>& data, int currentDepth, int& treeIndex);
		static std::vector<cv::Mat> predict(Tree* tree, std::vector<Patch*>& testData);
		static cv::Mat predictWithOnePatch(Node* node, Patch* patch);
	};

}