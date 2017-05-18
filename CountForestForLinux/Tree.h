#pragma once

// This file defines the Tree class, which is used to represent decision trees.
#include <iostream>
#include <string>
#include "Node.h"

namespace CrowdCount {
	class Tree
	{
	public:
		int decisionLevels_;

		Node* root;
		int number;
		int nonLeafNumber;
		int leafNumber;
		int hasPeopleLeafNumber;

		Tree()
		{
			number = 0; 
			nonLeafNumber = 0;
			leafNumber = 0;
			hasPeopleLeafNumber = 0;
		}
		void calTreeNodeAndPeopleCountInfo(Node* node);
		void printTreeInfo();
		void Serialize(const std::string& path);
		void Serialize(std::ostream& stream);

		static Tree* Deserialize(const std::string& path);
		static Tree* Deserialize(std::istream& i);
	};
}