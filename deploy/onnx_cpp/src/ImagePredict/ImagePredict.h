#define IMAGEPREDICT_EXPORTS
#ifdef IMAGEPREDICT_EXPORTS
#define IMAGEPREDICT_API __declspec(dllexport)
#else
#define IMAGEPREDICTL_API __declspec(dllimport)
#endif

#define _CRT_SECURE_NO_WARNINGS 
#pragma once

#include <windows.h>
#include <stdio.h>

#include <string>
#include <vector>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

class IMAGEPREDICT_API ImagePredict {
private:
	// Ort handle
	Ort::Env env_;
	Ort::SessionOptions session_ops_;
	Ort::Session* session_;

	// model info
	std::vector<char*> input_names_;
	std::vector<char*> output_names_;

	// HINSTANCE hdll;

	// inference results
	std::vector<std::array<float, 4>> bboxes_;
	std::vector<int> labels_;
	std::vector<cv::String> classes_name_;
	std::vector<float> scores_;
	std::vector<cv::Mat> masks_;

	// random colors list for mask show
	std::vector<cv::Scalar> colors_list_;

public:
	ImagePredict(bool with_gpu = false);
	~ImagePredict();
	bool LoadModel(char* model_path);
	bool PredictAction(cv::Mat& inputImg, float score_thresh = 0.7);
	cv::Mat ShowPredictMask(cv::Mat& inputImg, float scoreThreshold = 0.7);
	std::vector<cv::Mat> GetPredictMask();
	std::vector<int> GetPredictLabel();
	std::vector<float> GetPredictScore();

private:
	void softNMSBoxes_filter(float score_threshold = 0.5, float nms_threshold = 0.6);
	void drawDashRect(cv::Mat& img, cv::Point p1, cv::Point p2, 
				  cv::Scalar& color, int thickness = 1);
	cv::Scalar hsv_to_rgb(std::vector<float> hsv);
	std::vector<cv::Scalar> random_colors(int nbr, bool bright=true);
	Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims,
                             const Ort::MemoryInfo &memory_info_handler,
                             std::vector<float> &tensor_value_handler,
                             std::string data_format);

};