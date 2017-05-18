#include "Tree.h"
#include <fstream>

namespace CrowdCount {


	void Tree::calTreeNodeAndPeopleCountInfo(Node* node) {
		
		if (node) {
			//std::cout << "cao ni ma " << std::endl;
			this->number++;
			if (node->leaf) {
				this->leafNumber++;
				if (countNonZero(node->density)) {
					this->hasPeopleLeafNumber++;
				}
			}
			else {
				this->nonLeafNumber++;
			}
			calTreeNodeAndPeopleCountInfo(node->left);
			calTreeNodeAndPeopleCountInfo(node->right);
		}
		else {
			return;
		}
	

	}

	void Tree::printTreeInfo() {
		std::cout << "Node : " << this->number << std::endl;
		std::cout << "Non Leaf Node : " << this->nonLeafNumber << std::endl;
		std::cout << "Leaf Node : " << this->leafNumber << std::endl;
		std::cout << "Has people Leaf Node : " << this->hasPeopleLeafNumber << std::endl;
	}
	void Tree::Serialize(const std::string& path) {
		std::ofstream o(path.c_str(), std::ios_base::binary);
		Serialize(o);
	}
	void Tree::Serialize(std::ostream& stream) {

		this->root->Serialize(stream);
		
	}

	 Tree* Tree::Deserialize(const std::string& path) {
		 std::ifstream i(path.c_str(), std::ios_base::binary);
		 return Deserialize(i);
	}
	 Tree* Tree::Deserialize(std::istream& i) {
		Tree* t = new Tree();
		t->root= Node::Deserialize(i);
		return t;
	}
}