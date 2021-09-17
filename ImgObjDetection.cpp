#include "pch.h"
#include "ImgObjDetection.h"
#include <stdlib.h>
#include <malloc.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace std;
using namespace cv;
using namespace dnn;

vector<string> classes;		//儲存預設判斷類別
float confThreshold = 0.5;
float nmsThreshold  = 0.4;
int inpWidth  = 416;
int inpHeight = 416;

typedef std::uint64_t hash_t;
constexpr hash_t prime = 0x100000001B3ull;
constexpr hash_t basis = 0xCBF29CE484222325ull;

ImgObjDetection::ImgObjDetection()
{

}

ImgObjDetection::~ImgObjDetection()
{

}

hash_t hash_(char const* str)
{
	hash_t ret{ basis };
	while (*str)
	{
		ret ^= *str;
		ret *= prime;
		str++;
	}
	return ret;
}

constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis)
{
	return *str ? hash_compile_time(str + 1, (*str ^ last_value) * prime) : last_value;
}

void BoxColor(const char* name,Mat in, int left, int top, int right, int bottom)
{
	switch (hash_(name))
	{
	case hash_compile_time("person"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(127, 255, 0), 3);
		break;
	case hash_compile_time("bicycle"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(220, 220, 220), 3);
		break;
	case hash_compile_time("car"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(100, 149, 237), 3);
		break;
	case hash_compile_time("motorbike"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3);
		break;
	case hash_compile_time("aeroplane"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(47, 79, 79), 3);
		break;
	}
}

void BoxColor(const char* name, Mat in, int left, int top, int right, int bottom,int mod)
{
	switch (hash_(name))
	{
	case hash_compile_time("person"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(127, 255, 0), 3, mod);
		break;
	case hash_compile_time("bicycle"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(220, 220, 220), 3, mod);
		break;
	case hash_compile_time("car"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(100, 149, 237), 3, mod);
		break;
	case hash_compile_time("motorbike"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(0, 0, 255), 3, mod);
		break;
	case hash_compile_time("aeroplane"):
		rectangle(in, Point(left, top), Point(right, bottom), Scalar(47, 79, 79), 3, mod);
		break;
	}
}

void ImgObjDetection::TurnTo(unsigned char* in, unsigned char* out, int w, int h)
{
	int j = -1, k = h - 1;
	for (int i = 0; i < (h * w); i++)
	{
		j++;
		out[(3 * w * k) + (3 * j)]     = in[3 * i];
		out[(3 * w * k) + (3 * j) + 1] = in[(3 * i) + 1];
		out[(3 * w * k) + (3 * j) + 2] = in[(3 * i) + 2];
		if (j == w - 1 && k >= 0) { k--; j = -1; }
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
		BoxColor(classes[classId].c_str(), frame, left, top, right, bottom);
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5 * labelSize.height)), Point(left + round(1.5 * labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	BoxColor(classes[classId].c_str(), frame, left, top - round(1.5 * labelSize.height), left + round(1.5 * labelSize.width), top + baseLine, FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Get the names of the output layers
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

void ImgObjDetection::DetectImage(unsigned char* inptr,int w,int h, Modelstr str)
{
	ifstream ifs(str.classesFile.c_str());	//Load names of classes
	string line;
	while (getline(ifs, line))
		classes.push_back(line);

	//Load the network ( .cng & 權重Data寫入Dnn )
	Net net = readNetFromDarknet(str.modelConfiguration, str.modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_OPENCL);

	// Open a video file or an image file or a camera stream.
	int j = -1, k = h - 1;
	unsigned char* buf = new unsigned char[w * h * 3];
	TurnTo(inptr, buf, w, h);
	cv::Mat frame(h, w, CV_8UC3, buf);

	// Stop the program if reached end of video
	// Create a 4D blob from a frame.
	Mat blob;
	blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

	//Sets the input to the network
	net.setInput(blob);

	// Runs the forward pass to get output of the output layers
	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	// Remove the bounding boxes with low confidence
	postprocess(frame, outs);
	// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));
	TurnTo(buf, inptr, w, h);
	//imwrite("C:\\Users\\KYEC\\Desktop\\123.bmp", frame);
}

void ImgObjDetection::DetectVideo(string vstr, string ostr,Modelstr str)
{
	// Load names of classes
	ifstream ifs(str.classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// Load the network
	Net net = readNetFromDarknet(str.modelConfiguration, str.modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	// Open a video file or an image file or a camera stream.
	VideoCapture cap;
	VideoWriter video;
	Mat frame, blob;

	try
	{
		//Open the video file
		ifstream ifile(vstr);
		if (!ifile) throw("error");
		cap.open(vstr);
	}
	catch (...)
	{
		cout << "Could not open the input image/video stream" << endl;
		return;
	}

	// Get the video writer initialized to save the output video
	video.open(ostr,
		       VideoWriter::fourcc('M', 'J', 'P', 'G'), 
		       28, 
			   Size(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)));

	// Create a window
	static const string kWinName = "Yolo V3";
	namedWindow(kWinName, WINDOW_AUTOSIZE);

	// Process frames.
	while (waitKey(1) < 0)
	{
		// get frame from the video
		cap >> frame;

		// Stop the program if reached end of video
		if (frame.empty()) 
		{
			cout << "Done processing !!!" << endl;
			cout << "Output file is stored as " << ostr << endl;
			waitKey(3000);
			break;
		}
		// Create a 4D blob from a frame.
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		// Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		// Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		// Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		video.write(detectedFrame);
		imshow(kWinName, frame);
	}
	cap.release();
	video.release();
}

void ImgObjDetection::DetectStreaming(Modelstr str)
{
	// Load names of classes
	ifstream ifs(str.classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	//Load the network
	Net net = readNetFromDarknet(str.modelConfiguration, str.modelWeights);
	net.setPreferableBackend(DNN_BACKEND_DEFAULT);
	net.setPreferableTarget(DNN_TARGET_CPU);

	//Open a video file or an image file or a camera stream.
	VideoWriter video;
	Mat frame, blob;

	//開啟攝像頭
	cv::VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(CAP_PROP_FRAME_HEIGHT, 720);

	// Create a window
	static const string kWinName = "Yolo V3 Streaming";
	namedWindow(kWinName, WINDOW_AUTOSIZE);

	//Process frames.
	while (getWindowProperty(kWinName, WND_PROP_AUTOSIZE) >= 0)
	{
		//get frame from the video
		cap >> frame;

		//Stop the program if reached end of video
		if (frame.empty()) break;

		//Create a 4D blob from a frame.
		blobFromImage(frame, blob, 1 / 255.0, cvSize(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);

		//Sets the input to the network
		net.setInput(blob);

		//Runs the forward pass to get output of the output layers
		vector<Mat> outs;
		net.forward(outs, getOutputsNames(net));

		//Remove the bounding boxes with low confidence
		postprocess(frame, outs);

		//Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
		vector<double> layersTimes;
		double freq = getTickFrequency() / 1000;
		double t = net.getPerfProfile(layersTimes) / freq;
		string label = format("Inference time for a frame : %.2f ms", t);
		putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

		// Write the frame with the detection boxes
		Mat detectedFrame;
		frame.convertTo(detectedFrame, CV_8U);
		imshow(kWinName, frame);
		if (waitKey(30) >= 27) break;
	}
	destroyAllWindows();
	cap.release();
}