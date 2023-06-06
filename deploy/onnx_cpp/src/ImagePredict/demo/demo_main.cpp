#include "ModelPredict.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#ifdef WIN32
#include <windows.h>
#endif

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;
using namespace cv::dnn;

#ifdef WIN32
	#include <io.h>
	void readFileNameInDir(string strDir, 
						   vector<string>& vFileFullPath,
						   vector<string>& vFileName)
	{
		string file_dir = strDir;
		intptr_t handle;    // file handle
		struct _finddata_t fileInfo;    // file struct
		handle = _findfirst(strDir.append("/*").c_str(), &fileInfo);    // get file handle first
		while (!_findnext(handle, &fileInfo)){
			string filename = fileInfo.name;
			if (filename != "." && filename != ".."){
				vFileName.push_back(filename);
				vFileFullPath.push_back(file_dir.append("/") + filename);
			}
		}
		_findclose(handle);
	}
#else
	#include <dirent.h>
	void readFileNameInDir(string strDir, 
						   vector<string>& vFileFullPath, 
						   vector<string>& vFileName)
	{
		struct dirent* pDirent;
		DIR* pDir = opendir(strDir.c_str());
		if (pDir != NULL)
		{
			while ((pDirent = readdir(pDir)) != NULL)
			{
				string strFileName = pDirent->d_name;
				
				if (strFileName != "." && strFileName != ".."){
					string strFileFullPath = strDir + "/" + strFileName;
					vFileName.push_back(strFileName);
					vFileFullPath.push_back(strFileFullPath);
				}
				
			}
		}
	}
#endif

int main(int argc, char* argv[])
{
	char* model_path = "../../models/sack/patched.onnx";
	String img_dir = "../../images/sack/";

	string save_dir = "../../inference_results/sack";
	vector<string> vec_img_paths, vec_img_names;
	readFileNameInDir(img_dir, vec_img_paths, vec_img_names);

	// construct ModelPredict object and load model
	ModelPredict onnx_ip(false);
	onnx_ip.LoadModel(model_path);

	cout << "INFO: All inference images: " << vec_img_paths.size() << endl;
	for (size_t i = 0; i < vec_img_paths.size(); i++){
		cout << "INFO: inference at img: " << vec_img_names[i] << endl;
		cv::Mat img = imread(vec_img_paths[i]);

		float score_thresh = 0.6;
		bool status = onnx_ip.PredictAction(img, score_thresh);

		cv::Mat result_img = onnx_ip.ShowPredictMask(img, score_thresh);

		cv::String save_path = save_dir + "/" + vec_img_names[i];
		imwrite(save_path, result_img);
		cv::imshow("inference result", result_img);
		waitKey(0);
	}
	return 0;
}


