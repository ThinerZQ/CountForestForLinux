#pragma once
#include "Frame.h"
#include "Patch.h"

namespace CrowdCount{
	class Utils{
	public:
		static void load(const std::vector<std::string>& originimgs, const std::vector<std::string>&  dottedimgs,const std::string&  roipath, const std::string&  pmappath, std::vector<Patch*>& patches);
		static void load(const std::string& originimg, const std::string& dottedimg,cv::Mat& roi,cv::Mat& pmap, std::vector<Patch*>& patches);
		static void saveResult(const std::string& path, const std::string& dmapSavePath,std::vector<int>& predictSequence,std::vector<float>& predictNums,std::vector<float>& groundTruthNums, std::vector<Frame*>& densityMaps);
		static cv::Mat loadROI(const std::string& roipath);
		static cv::Mat loadPmap(const std::string& pmappath);
	};


}