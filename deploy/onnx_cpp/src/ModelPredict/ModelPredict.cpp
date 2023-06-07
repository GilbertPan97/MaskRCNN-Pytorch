#include "ModelPredict.h"

#include <fstream>
#ifdef WIN32
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>

using namespace std;
using namespace cv;

ModelPredict::ModelPredict(bool gpu){
    // load onnxruntime dll (CPU or GPU version)
	// cout << "INFO: Start load onnxruntime.dll";
	// this->hdll = LoadLibrary("onnxruntime.dll");
	// if (this->hdll == NULL) {
	// 	cout << "ERROR: Fail to load onnxruntime.dll";
	// 	FreeLibrary(this->hdll);
	// 	throw("Load onnxruntime.dll fail.");
	// }

    // initial ort env and session options handle
    env_ = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "OnnxModel");
	session_ops_.SetIntraOpNumThreads(1);	// op thread
    session_ops_.SetGraphOptimizationLevel(
		GraphOptimizationLevel::ORT_ENABLE_ALL);	// Enable all possible optimizations

#if WITH_GPU == true
	if (!gpu){
		std::cout << "WARNING: GPU option is not selected, model are running at cpu.\n";
	}
	else{
		try{
			// Model acceleration with gpu-0
			OrtSessionOptionsAppendExecutionProvider_CUDA(session_ops_, 0);
		}
		catch (const Ort::Exception& exception) {
			std::cerr << "Error: " << exception.what() << std::endl;
			throw std::runtime_error("ERROR: No cuda detected, run model on gpu fail.\n");
		}
	}	
#endif

	// cout << "INFO:  -- load InitModel";
	// this->init_model = (InitModel)GetProcAddress(this->hdll, "InitModel");
	// if (this->init_model == NULL) {
	// 	cout << "INFO: Fail to load InitModel";
	// 	FreeLibrary(this->hdll);
	// 	throw("Initial model fail.");
	// }
	// cout << "INFO: Succeed initing ModelPredict";
}

ModelPredict::~ModelPredict(){
	cout << "INFO: Destruct model";
	delete session_;
}

bool ModelPredict::LoadModel(char* model_path){
    // creat session to load  model.
	cout << "INFO: Start loading model." << endl;

#ifdef WIN32
	int bufSize = MultiByteToWideChar(CP_ACP, 0, model_path, -1, NULL, 0);
	wchar_t* w_model_path = new wchar_t[bufSize];
	MultiByteToWideChar(CP_ACP, 0, model_path, -1, w_model_path, bufSize);
	// Create model session and load model
	try {
		session_ = new Ort::Session(env_, w_model_path, session_ops_);
	}
	catch (const Ort::Exception& exception) {
		std::cerr << "Error: " << exception.what() << std::endl;
		return false;
	}
#else
	try {
		session_ = new Ort::Session(env_, model_path, session_ops_);
	}
	catch (const Ort::Exception& exception) {
		std::cerr << "Error: " << exception.what() << std::endl;
		return false;
	}
#endif

	// print model input layer (node names, types, shape etc.)
    size_t num_input_nodes = session_->GetInputCount();
	size_t num_output_nodes = session_->GetOutputCount();
	Ort::AllocatorWithDefaultOptions allocator;
	for (size_t i = 0; i < num_input_nodes; i++){
		auto input_name_Ptr = session_->GetInputName(i, allocator);
		input_names_.push_back(input_name_Ptr);
		cout << "INFO: Model input name-[" << i << "] is: " 
			<< input_names_[i] << endl;
	}
	for (size_t i = 0; i < num_output_nodes; i++){
		auto output_name_Ptr = session_->GetOutputName(i, allocator);
		output_names_.push_back(output_name_Ptr);
		cout << "INFO: Model output name-[" << i << "] is: "
			<< output_names_[i] << endl;
	}
    
	cout << "INFO: Succeed loading model";
	return true;
}

bool ModelPredict::PredictAction(cv::Mat& inputImg, float score_thresh){ 
	// clear results member
    bboxes_.clear();
	labels_.clear();
	scores_.clear();
	masks_.clear();

    // input_dims {batch_size:1, chanel:3, height: ,width: }
	auto input_dims = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	auto output_dims = session_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

    // reshape input image accroding to model input size
    if (input_dims[2] != inputImg.rows || input_dims[3] != inputImg.cols){
		Size in_size(input_dims[3], input_dims[2]);		// width(cols), height(rows)
		resize(inputImg, inputImg, in_size);
	}

	// construct input image tensor
	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	std::vector<float> tensor_value;
	std::vector<Ort::Value> input_tensors;
	input_tensors.push_back(create_tensor(inputImg, input_dims, memory_info, tensor_value, "CHW"));

	// inference run
	double timeStart = (double)getTickCount();
	auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names_.data(), 
		input_tensors.data(), input_tensors.size(), output_names_.data(), output_names_.size());
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "Inference time consume : " << nTime  << " s."<< endl;
	
	// Allocate outputData from output tensors ptr
	if (output_tensors.size() != 4){
		exit(-1);
	}

	using DataOutputType = std::pair<float*, std::vector<int64_t>>;
	std::vector<DataOutputType> outputData;
	outputData.reserve(4);

	for (auto& elem : output_tensors) {
		outputData.emplace_back(std::make_pair(std::move(elem.GetTensorMutableData<float>()),
			elem.GetTensorTypeAndShapeInfo().GetShape()));
	}

	// Get Outputs and check them
	size_t nBoxes = outputData[1].second[0];

	int mask_height = outputData[3].second[2];
	int mask_width = outputData[3].second[3];

	for (size_t i = 0; i < nBoxes; ++i) {
		// get bound box
		float xmin = outputData[0].first[i * 4 + 0];
		float ymin = outputData[0].first[i * 4 + 1];
		float xmax = outputData[0].first[i * 4 + 2];
		float ymax = outputData[0].first[i * 4 + 3];
		bboxes_.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});

		// get classes indix
		labels_.emplace_back(reinterpret_cast<int64_t*>(outputData[1].first)[i]);

		// get predice scores
		scores_.emplace_back(outputData[2].first[i]);

		// get mask
		cv::Mat curMask(mask_height, mask_width, CV_32FC1);
		memcpy(curMask.data, outputData[3].first + i * mask_height * mask_width, 
			mask_height * mask_width * sizeof(float));
		masks_.emplace_back(curMask);
	}

	// get random colors list for mask show
	colors_list_ = random_colors(nBoxes);

	// filter predict results with soft NMS
	softNMSBoxes_filter();
	return true;
}

cv::Mat ModelPredict::ShowPredictMask(cv::Mat& inputImg, float scoreThreshold){
    assert(bboxes_.size() == labels_.size());

    cv::Mat result = inputImg.clone();
	if (bboxes_.size() == 0)
		return result;
	
	// filter low socre results (filter seq: n-1->0)
	size_t nbr_result = bboxes_.size();
	for (size_t i = nbr_result; i > 0; i--){
		if (scores_[i-1] < scoreThreshold){
			bboxes_.erase(std::begin(bboxes_)+i-1);
			masks_.erase(std::begin(masks_)+i-1);
			labels_.erase(std::begin(labels_)+i-1);
			scores_.erase(std::begin(scores_)+i-1);
		}else
			continue;
	}
	
	float maskThreshold = 0.5;
	// -----------------------Visualize masks-----------------------//
    for (size_t i = 0; i < bboxes_.size(); ++i) {
		cv::Scalar& curColor = colors_list_[i];
        cv::Mat curMask = masks_[i].clone();

        cv::Mat filterMask = (curMask > maskThreshold);		// filter mask data

        cv::Mat colored_img = (0.3 * curColor + 0.7 * result);	// splash transparent color to image
        colored_img.convertTo(colored_img, CV_8UC3);

        std::vector<cv::Mat> contours;
        cv::Mat hierarchy;
        filterMask.convertTo(filterMask, CV_8U);

		// draw mask contour on colored image
        cv::findContours(filterMask, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(colored_img, contours, -1, curColor, 2, cv::LINE_8, hierarchy, 100);
        colored_img.copyTo(result, filterMask);		// copy colored mask region to result
    }
	// -----------------------Draw bbox and labels-----------------------//
	for (size_t i = 0; i < bboxes_.size(); ++i) {
		// read color from colors list
		cv::Scalar& curColor = colors_list_[i];

		// get label name
		uint64_t classIdx = labels_[i];
		string class_name;
		if (classes_name_.size()!=0)
			class_name = classes_name_[classIdx-1];
		else
			class_name = "target";

		// get score and transfer to string
		float score = scores_[i];
		string str_score = to_string(score).substr(0, to_string(score).find(".") + 4);

		// marker (class name and predict score)
		cv::String marker = class_name + " " + str_score;
		
		// draw bbox
		auto& curBbox = bboxes_[i];
		drawDashRect(result, cv::Point(curBbox[0], curBbox[1]), cv::Point(curBbox[2], curBbox[3]), curColor, 2);
		
		// draw marker (class name and score)
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(marker, cv::FONT_HERSHEY_COMPLEX, 0.35, 1, &baseLine);
        cv::putText(result, marker, cv::Point(curBbox[0], curBbox[1] - labelSize.height), 
					cv::FONT_ITALIC, 0.6, curColor, 2);
	}

    return result;
}

void ModelPredict::softNMSBoxes_filter(float score_threshold, float nms_threshold){
	// data transfer
	std::vector<float> updated_scores;
	std::vector<cv::Rect> cv_boxes;
	for (size_t i = 0; i < bboxes_.size(); i++){
		int left = bboxes_[i][0];
		int top = bboxes_[i][1];
		int width = bboxes_[i][2]-bboxes_[i][0];
		int heigh = bboxes_[i][3]-bboxes_[i][1];
		cv_boxes.push_back(Rect(left, top, width, heigh));
	}
	
	// filter NMS results
	std::vector<int> perf_indx;
	cv::dnn::softNMSBoxes(cv_boxes, scores_, updated_scores,
						  score_threshold, nms_threshold, perf_indx);

	// reshape NMS result
	int ori_nbbox = bboxes_.size();		// original number of bboxes
	scores_ = updated_scores;
	for (size_t i = 0; i < perf_indx.size(); i++){
		int new_id = perf_indx[i];
		std::array<float, 4> box = {bboxes_[new_id][0], bboxes_[new_id][1], 
								    bboxes_[new_id][2], bboxes_[new_id][3]};
		bboxes_.push_back(box);		// new results push back at original results
		labels_.push_back(labels_[new_id]);
		masks_.push_back(masks_[new_id]);
	}
	bboxes_.erase(bboxes_.begin(), bboxes_.begin()+ori_nbbox);		// erase original results
	labels_.erase(labels_.begin(), labels_.begin()+ori_nbbox);
	masks_.erase(masks_.begin(), masks_.begin()+ori_nbbox);
}

std::vector<cv::Mat> ModelPredict::GetPredictMask()
{
	return masks_;
}

std::vector<int> ModelPredict::GetPredictLabel()
{
	return labels_;
}

std::vector<float> ModelPredict::GetPredictScore()
{
	return scores_;
}



cv::Scalar ModelPredict::hsv_to_rgb(std::vector<float> hsv){
	// hsv value convert to rgb (0~255)
	float alpha = 0.5;

	cv::Scalar rgb;
	float h = hsv[0], s = hsv[1], v = hsv[2];
	if(s == 0.0){
		rgb = {v, v, v};
		return rgb;
	}
    int i = int(h*6.0); // XXX assume int() truncates!
	float f = (h*6.0) - i;
    float p = v*(1.0 - s);
    float q = v*(1.0 - s*f);
    float t = v*(1.0 - s*(1.0-f));
	i = i%6;

	// get rgb (value range: 0~1)
	switch (i)
	{
	case 0:
		rgb[0] = v;
		rgb[1] = t;
		rgb[2] = p;
		break;
	case 1:
		rgb[0] = q;
		rgb[1] = v;
		rgb[2] = p;
		break;
	case 2:
		rgb[0] = p;
		rgb[1] = v;
		rgb[2] = t;
		break;
	case 3:
		rgb[0] = p;
		rgb[1] = q;
		rgb[2] = v;
		break;	
	case 4:
		rgb[0] = t;
		rgb[1] = p;
		rgb[2] = v;
		break;	
	case 5:
		rgb[0] = v;
		rgb[1] = p;
		rgb[2] = q;
		break;
	default:
		break;
	}

	// transfer rgb (value range: 0~255)
	for (size_t i = 0; i < 3; i++){
		rgb[i] = rgb[i] * 255;
	}

	return rgb;
}

void ModelPredict::drawDashRect(cv::Mat& img, cv::Point p1, cv::Point p2, 
				  cv::Scalar& color, int thickness){
	int w = p2.x - p1.x;		// width
	int h = p2.y - p1.y;		// height
 
	int tl_x = p1.x;	// top left x
	int tl_y = p1.y;	// top left y
 
	int linelength = 4, dashlength = 6;
    int totallength = dashlength + linelength;
	int nCountX = w/totallength;	//
	int nCountY = h/totallength;	//
 
	cv::Point start, end;		// start and end point of each dash
 
	// draw the horizontal lines
	start.y = tl_y;
	start.x = tl_x;
 
	end.x = tl_x;
	end.y = tl_y;
	
	for (int i=0; i<nCountX; i++){
		end.x=tl_x+(i+1)*totallength-dashlength;	// draw top dash line
		end.y=tl_y;
		start.x=tl_x+i*totallength;
		start.y=tl_y;
		cv::line(img, start, end, color, thickness);   
	}

	for (int i=0;i<nCountX;i++){  
		start.x=tl_x+i*totallength;
		start.y=tl_y+h;
		end.x=tl_x+(i+1)*totallength-dashlength;	//draw bottom dash line
		end.y=tl_y+h;
		cv::line(img, start, end, color, thickness);     
	}
 
	for (int i=0;i<nCountY;i++){  
		start.x=tl_x;
		start.y=tl_y+i*totallength;
		end.y=tl_y+(i+1)*totallength-dashlength;	//draw left dash line
		end.x=tl_x;
		cv::line(img, start, end, color, thickness);     
	}
 
	for (int i=0;i<nCountY;i++){  
		start.x=tl_x+w;
		start.y=tl_y+i*totallength;
		end.y=tl_y+(i+1)*totallength-dashlength;	//draw right dash line
		end.x=tl_x+w;
		cv::line(img, start, end, color, thickness);     
	}
 
}

std::vector<cv::Scalar> ModelPredict::random_colors(int nbr, bool bright){
	// Generate random colors.
    // To get visually distinct colors, generate them in HSV space then
    // convert to RGB.
	float brightness = 1.0;
	if (!bright)
		brightness = (float) 0.7;
	
	std::vector<cv::Scalar> list_colors;
	for (size_t i = 0; i < nbr; i++){
		std::vector<float> hsv = {i/float(nbr), 1, brightness};
		list_colors.push_back(hsv_to_rgb(hsv));
	}
    return list_colors;
}

Ort::Value ModelPredict::create_tensor(const cv::Mat &mat,
                                       const std::vector<int64_t> &tensor_dims,
                                       const Ort::MemoryInfo &memory_info_handler,
                                       std::vector<float> &tensor_value_handler,
                                       std::string data_format){

	const unsigned int rows = mat.rows;
  	const unsigned int cols = mat.cols;
	const unsigned int channels = mat.channels();

	cv::Mat mat_ref;
	if (mat.type() != CV_32FC(channels)) 
		mat.convertTo(mat_ref, CV_32FC(channels));
	else 
		mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels
	
	cv::Mat norm_mat_ref;
	cv::cvtColor(mat_ref, mat_ref, cv::COLOR_BGR2RGB);	
	normalize(mat_ref, norm_mat_ref, 1.0, 0, NORM_MINMAX);	// Normalize the pixel value space

	if (tensor_dims.size() != 4) 
		throw std::runtime_error("dims mismatch.");
	if (tensor_dims.at(0) != 1) 
		throw std::runtime_error("batch != 1");

	// CXHXW
	if (data_format == "CHW"){
		const unsigned int target_channel = tensor_dims.at(1);
		const unsigned int target_height = tensor_dims.at(2);
		const unsigned int target_width = tensor_dims.at(3);
		const unsigned int target_tensor_size = target_channel * target_height * target_width;

		if (target_channel != channels)
			throw std::runtime_error("channel mismatch.");
		tensor_value_handler.resize(target_tensor_size);

		cv::Mat resize_mat_ref;
		if (target_height != rows || target_width != cols)
			cv::resize(norm_mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
		else
			resize_mat_ref = norm_mat_ref; 	// reference only. zero-time cost.

		std::vector<cv::Mat> mat_channels;
		cv::split(resize_mat_ref, mat_channels);		// mat_channels: R->G->B

		// CXHXW transform
		for (unsigned int i = 0; i < channels; ++i)
		std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
					mat_channels.at(i).data, 
					target_height * target_width * sizeof(float));
			
		return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
											target_tensor_size, tensor_dims.data(),
											tensor_dims.size());
	}
	// HXWXC
	const unsigned int target_height = tensor_dims.at(1);
	const unsigned int target_width = tensor_dims.at(2);
	const unsigned int target_channel = tensor_dims.at(3);
	const unsigned int target_tensor_size = target_channel * target_height * target_width;

	if (target_channel != channels) 
		throw std::runtime_error("channel mismatch!");
	tensor_value_handler.resize(target_tensor_size);

	cv::Mat resize_mat_ref;
	if (target_height != rows || target_width != cols)
		cv::resize(norm_mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
	else 
		resize_mat_ref = norm_mat_ref; 	// reference only. zero-time cost.

	std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, 
		target_tensor_size * sizeof(float));

	return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
										   target_tensor_size, tensor_dims.data(),
										   tensor_dims.size());
}