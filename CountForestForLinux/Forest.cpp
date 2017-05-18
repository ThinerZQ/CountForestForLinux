#include "Forest.h"
namespace CrowdCount {


	void Forest::AddTree(Tree* tree)
	{
		trees.push_back(tree);
	}
	const Tree& Forest::GetTree(int index) const
	{
		return *trees[index];
	}
	Tree& Forest::GetTree(int index)
	{
		return *trees[index];
	}
	int Forest::TreeCount() const
	{
		return trees.size();
	}


	void Forest::Serialize(const std::string& path)
	{
		std::cout << "save model to :" << path << std::endl;
		std::ofstream o(path.c_str(), std::ios_base::binary);
		Serialize(o);
	}

	void Forest::Serialize(std::ostream& stream)
	{
		int treeCount = TreeCount();
		stream.write((const char*)(&treeCount), sizeof(treeCount));

		for (int t = 0; t < TreeCount(); t++) {
			GetTree((t)).Serialize(stream);
			std::cout << "Tree " << t << "info : " << std::endl;
			GetTree(t).calTreeNodeAndPeopleCountInfo(GetTree(t).root);
			GetTree(t).printTreeInfo();
		}

		if (stream.bad())
			throw std::runtime_error("Forest serialization failed.");
	}


	 Forest* Forest:: Deserialize(const std::string& path)
	{
		std::ifstream i(path.c_str(), std::ios_base::binary);

		return Forest::Deserialize(i);
	}

	 Forest*  Forest::Deserialize(std::istream& i)
	{
		 Forest* forest = new Forest();
		 int treeCount = 0;
		 i.read((char*)&treeCount, sizeof(treeCount));

		 
		 for (int t = 0; t < treeCount; t++)
			 forest->AddTree(Tree::Deserialize(i));
		 return forest;
	}




}
