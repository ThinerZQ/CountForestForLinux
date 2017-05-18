#pragma once
#include <iostream>
namespace CrowdCount {
	using std::cout;
	using std::endl;
	class TrainingParameters
	{
	public:
		TrainingParameters()
		{
			// Some sane defaults will need to be changed per application.
			NumberOfTrees = 2;
			NumberOfCandidateFeatures = 10;
			NumberOfCandidateThresholdsPerFeature = 10;
			MaxDecisionLevels = 11;
			MinSampleSize = 20;
			MaxDepthUpper = 8;
			patchsize = 13;
			featureDimension = patchsize * patchsize * 5;
			patchNum = 1000;
			stride = 5;
		} 

		void print() {
		
			cout << "predict parameters : " << endl <<
				"\t treeNumber = " << NumberOfTrees << endl <<
				"\t NumberOfCandidateFeatures = " << NumberOfCandidateFeatures << endl <<
				"\t NumberOfCandidateThresholdsPerFeature = " << NumberOfCandidateThresholdsPerFeature << endl <<
				"\t MaxDecisionLevels = " << MaxDecisionLevels << endl <<
				"\t MinSampleSize = " << MinSampleSize << endl <<
				"\t MaxDepthUpper = " << MaxDepthUpper << endl <<
				"\t patchsize = " << patchsize << endl <<
				"\t featureDimension = " << featureDimension << endl <<
				"\t patchNum = " << patchNum <<endl<<
				"\t height = " << height << endl <<
				"\t width = " << width << endl;
		}

		 int NumberOfTrees; 
		 int NumberOfCandidateFeatures;
		 int NumberOfCandidateThresholdsPerFeature;
		 int MaxDecisionLevels;
		 int MinSampleSize;
		 int MaxDepthUpper;
		 int patchsize;
		 int featureDimension;
		 int patchNum;
		 int postivePatchNum;
		 int height;
		 int width;
		 int stride;
	};
	extern TrainingParameters* parameter;
}