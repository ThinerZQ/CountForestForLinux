#pragma once

#include <fstream>
#include <iostream>
#include <vector>
#include "Tree.h"

namespace CrowdCount {
	
	class Tree;
	class Forest 
	{

		std::vector<Tree* > trees;

	public:
		typedef  std::vector< Tree>::size_type TreeIndex;

		~Forest()
		{
			for (TreeIndex t = 0; t < trees.size(); t++)
				delete trees[t];
		}

		void AddTree(Tree* tree);
		const Tree& GetTree(int index) const;
		Tree& GetTree(int index);
		int TreeCount() const;


		void Serialize(const std::string& path);
		void Serialize(std::ostream& stream);

		static Forest*  Deserialize(const std::string& path);
		static Forest*  Deserialize(std::istream& i);
		
	
		
	};

}
