#include "DensityEstimation.h"
#include "RandomForest.h"
#include "Patch.h"
#include <iostream>
namespace CrowdCount{
	
	 Forest* DensityEstimation::TrainForest(const std::vector<Patch*>& trainingData){
		std::cout << "Running training forest..." << std::endl;

		Forest* forest = RandomForest::train(trainingData);
		return forest;
	}
}
