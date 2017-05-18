#include "RandomForest.h"
#include "DecisionTree.h"
#include "TrainingParameters.h"
namespace CrowdCount {

	 Forest* RandomForest::train(const std::vector<Patch*>& data)
	{

		
		Forest* forest = new Forest();
		srand((unsigned)time(0));
#pragma omp parallel for num_threads(parameter->NumberOfTrees)
		for (int t = 0; t < parameter->NumberOfTrees; t++)
		{
			std::vector<Patch*> currentData;
			int index = 0; 
			for (int i = 0; i < data.size(); i++)
			{
				index = rand() % data.size();
				currentData.push_back(data[i]);
			}
			
			Tree* tree = DecisionTree::train(currentData,t);
#pragma omp critical
			forest->AddTree(tree);
		}


		return forest;
	}
	 void RandomForest::predict(Forest* forest, vector<Patch*>& testData)
	{
#pragma omp parallel for num_threads(10)
		for (int i = 0; i < testData.size(); i++) {
			if (testData[i]->hasForeground) {
				vector<cv::Mat> allPredictLabel4Patch;
				//计算一个Patch在每一棵树上的密度
				for (int t = 0; t < forest->TreeCount(); t++) {
					cv::Mat predictLabel = DecisionTree::predictWithOnePatch(forest->GetTree(t).root, testData[i]);
					allPredictLabel4Patch.push_back(predictLabel);
				}
				//计算一个Patch的平均密度
				cv::Mat avgDensity = cv::Mat::zeros(parameter->patchsize, parameter->patchsize, CV_32F);
				for (int j = 0; j < allPredictLabel4Patch.size(); j++)
				{
					cv::add(avgDensity, allPredictLabel4Patch[j], avgDensity);
				}
		

				for (int r = 0; r < avgDensity.rows; r++)
				{
					for (int c = 0; c < avgDensity.cols; c++)
						avgDensity.at<float>(r, c) = avgDensity.at<float>(r, c) / allPredictLabel4Patch.size();
				}

				testData[i]->predictLabel = avgDensity;

			/*	if(countNonZero(testData[i]->label)>=5)
					cout << countNonZero(testData[i]->label)<< endl;
*/


			}
			else {
				//没有前景像素点
				testData[i]->predictLabel = cv::Mat::zeros(parameter->patchsize, parameter->patchsize, CV_32F);
			}
		}
	}


}