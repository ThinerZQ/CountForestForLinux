#pragma once
#include <vector>
namespace CrowdCount {
	using namespace std;
	class Patch;
	class Forest;
	class RandomForest{
		
	public:
		static Forest* train(const std::vector<Patch*>& data);
		static void predict(Forest* forest, vector<Patch*>& testData);
};
}