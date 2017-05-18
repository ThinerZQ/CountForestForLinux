#pragma once
#include <vector>
namespace CrowdCount{
	class Forest;
	class Patch;
	class DensityEstimation
	{
	public:
		static Forest* TrainForest(const std::vector<Patch*>& trainingData);
	};

}