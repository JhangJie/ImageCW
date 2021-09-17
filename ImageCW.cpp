#include "pch.h"
#include "ImgProcessing.h"
#include "ImageCW.h"
#include <stdlib.h>
#include <malloc.h>
#include <mmsystem.h>
#include "math.h"
#include <direct.h>
#include <ctype.h>
#include <io.h>
#include <iostream>
#include <fstream>
#include <string> 
#include <sstream>
#include <vector>
#include <time.h>
using namespace std;

ImageProcessing ImgPr;

bool loadImageFile(const char* pstr)
{
	FILE* bfile;
	bfile = fopen(pstr, "rb");
	bool Flag = ImgPr.Load(bfile);
	fclose(bfile);
	return Flag;
}

bool saveImageFile(const char* pstr)
{
	FILE* bfile;
	bfile = fopen(pstr, "wb");
	bool Flag = ImgPr.Save(bfile);
	fclose(bfile);
	return Flag;
}

bool getValue_INT(int x, int y, int* val)
{
	return ImgPr.GetValue(x, y, val);
}

bool getValue_FLO(int x, int y, float* val)
{
	return ImgPr.GetValue(x, y, val);
}

bool putValue_INT(int x, int y, int* val)
{
	return ImgPr.PutValue(x, y, val);
}

bool putValue_FLO(int x, int y, float* val)
{
	return ImgPr.PutValue(x, y, val);
}

bool imgMean(double* dmean)
{
	return ImgPr.ImgMean(dmean);
}

bool imgSTDEV(double* dstdev)
{
	return ImgPr.ImgSTDEV(dstdev);
}

bool imgHsum(double* dhsum)
{
	return ImgPr.ImgHsum(dhsum);
}

bool imgVsum(double* dvsum)
{
	return ImgPr.ImgVsum(dvsum);
}

bool imgMaxMin(double* dmax, double* dmin)
{
	return ImgPr.ImgMaxMin(dmax, dmin);
}

bool addImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.AddImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool addImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.AddImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool subtractImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.SubtractImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool subtractImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.SubtractImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool multiplyImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.MultiplyImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool multiplyImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.MultiplyImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool divideImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.DivideImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool divideImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.DivideImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool andImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.AndImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool andImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.AndImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool orImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.OrImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool orImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.OrImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool xorImage(const char* str1, const char* str2, const char* stro)
{
	if ((_access(str1, F_OK)) == -1 || (_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img1, img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str1, "rb");
	bool Flag = img1.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	bfile = fopen(str2, "rb");
	Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &img1.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &img1.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.XorImage(img1.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool xorImage_(const char* str2, const char* stro)
{
	if ((_access(str2, F_OK)) == -1) return false;
	if (stro == "") return false;
	ImageProcessing img2, imgo;

	FILE* bfile = NULL;
	bfile = fopen(str2, "rb");
	bool Flag = img2.Load(bfile);
	fclose(bfile);
	if (!Flag) return false;

	memcpy(&imgo.ImagInfo.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(imgo.ImagInfo.BmpFile));
	memcpy(&imgo.ImagInfo.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(imgo.ImagInfo.BmpInfo));
	ImgPr.XorImage(ImgPr.ImagInfo, img2.ImagInfo, &imgo.ImagInfo);
	bfile = fopen(stro, "wb");
	Flag = imgo.Save(bfile);
	fclose(bfile);
	return true;
}

bool addValue_INT(int x, int y, int* val)
{
	return ImgPr.AddValue(&ImgPr.ImagInfo, x, y, val);
}

bool addValue_FLO(int x, int y, float* val)
{
	return ImgPr.AddValue(&ImgPr.ImagInfo, x, y, val);
}

bool subtractValue_INT(int x, int y, int* val)
{
	return ImgPr.SubtractValue(&ImgPr.ImagInfo, x, y, val);
}

bool subtractValue_FLO(int x, int y, float* val)
{
	return ImgPr.SubtractValue(&ImgPr.ImagInfo, x, y, val);
}

bool multiplyValue_INT(int x, int y, int* val)
{
	return ImgPr.MultiplyValue(&ImgPr.ImagInfo, x, y, val);
}

bool multiplyValue_FLO(int x, int y, float* val)
{
	return ImgPr.MultiplyValue(&ImgPr.ImagInfo, x, y, val);
}

bool divideValue_INT(int x, int y, int* val)
{
	return ImgPr.DivideValue(&ImgPr.ImagInfo, x, y, val);
}

bool divideValue_FLO(int x, int y, float* val)
{
	return ImgPr.DivideValue(&ImgPr.ImagInfo, x, y, val);
}

bool notImage()
{
	return ImgPr.NotImage(&ImgPr.ImagInfo);
}

bool lShiftImage(int Lsft)
{
	return ImgPr.LShiftImage(&ImgPr.ImagInfo, Lsft);
}

bool rShiftImage(int Rsft)
{
	return ImgPr.RShiftImage(&ImgPr.ImagInfo, Rsft);
}

bool andValue(int x, int y, int* val)
{
	return ImgPr.AndValue(&ImgPr.ImagInfo, x, y, val);
}

bool orValue(int x, int y, int* val)
{
	return ImgPr.OrValue(&ImgPr.ImagInfo, x, y, val);
}

bool xorValue(int x, int y, int* val)
{
	return ImgPr.XorValue(&ImgPr.ImagInfo, x, y, val);
}

bool setMask(int mask)
{
	return ImgPr.SetMask(mask);
}

bool reverse()
{
	return ImgPr.Reverse(ImgPr.ImagInfo);
}

bool grayscale()
{
	return ImgPr.Grayscale(ImgPr.ImagInfo);
}

bool binarization()
{
	return ImgPr.Binarization(ImgPr.ImagInfo);
}

bool rGrayscale()
{
	return ImgPr.RGrayscale(ImgPr.ImagInfo);
}

bool histogram(double* histdata, int aisle, int shift)
{
	return ImgPr.Histogram(ImgPr.ImagInfo, histdata, aisle, shift);
}

bool blurBoxFilter()
{
	return ImgPr.Blur_BoxFilter(ImgPr.ImagInfo);
}

bool blurGaussianFilter()
{
	return ImgPr.Blur_GaussianFilter(ImgPr.ImagInfo);
}

bool blurMotion()
{
	return ImgPr.Blur_Motion(ImgPr.ImagInfo);
}

bool sharpNess()
{
	return ImgPr.Sharpness(ImgPr.ImagInfo);
}

bool edgeDetectionSobelfilter()
{
	return ImgPr.EdgeDetection_Sobelfilter(ImgPr.ImagInfo);
}

bool edgeDetectionLaplacianEdgeDetector()
{
	return ImgPr.EdgeDetection_LaplacianEdgeDetector(ImgPr.ImagInfo);
}

bool embossing()
{
	return ImgPr.Embossing(ImgPr.ImagInfo);
}

bool medianFilter()
{
	return ImgPr.MedianFilter(ImgPr.ImagInfo);
}

bool fuzzyMedianFilter()
{
	return ImgPr.FuzzyMedianFilter(ImgPr.ImagInfo);
}

bool prewittFilter()
{
	return ImgPr.PrewittFilter(ImgPr.ImagInfo);
}

bool loGFilter()
{
	return ImgPr.LoGFilter(ImgPr.ImagInfo, 1, 9);
}

bool doGFilter()
{
	return ImgPr.DoGFilter(ImgPr.ImagInfo, 0.7, 0.8, 9);
}

bool imgRange(float* kel, int x, int y)
{
	IMAGEINFOR out;
	int Square = sqrt(1089);
	int n_w = Square;
	int n_h = Square;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.width = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	return ImgPr.ImgRange(ImgPr.ImagInfo, out, kel, x, y);
}

bool templateMatching(float* kel, int mod)
{
	return ImgPr.TemplateMatching(ImgPr.ImagInfo, kel, mod);
}

bool turnto()
{
	return ImgPr.Turnto(ImgPr.ImagInfo);
}

bool conversion()
{
	return ImgPr.Conversion(ImgPr.ImagInfo);
}

bool stretch(int mul)
{
	IMAGEINFOR out;
	int n_w = ImgPr.ImagInfo.BmpInfo.width * mul;
	int n_h = ImgPr.ImagInfo.BmpInfo.height;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Stretch(ImgPr.ImagInfo, out, mul);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool pullup(int mul)
{
	IMAGEINFOR out;
	int n_w = ImgPr.ImagInfo.BmpInfo.width;
	int n_h = ImgPr.ImagInfo.BmpInfo.height * mul;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Pullup(ImgPr.ImagInfo, out, mul);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool narrowing(int mul)
{
	IMAGEINFOR out;
	int n_w = ImgPr.ImagInfo.BmpInfo.width / mul;
	int n_h = ImgPr.ImagInfo.BmpInfo.height;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Narrowing(ImgPr.ImagInfo, out, mul);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool shrink(int mul)
{
	IMAGEINFOR out;
	int n_w = ImgPr.ImagInfo.BmpInfo.width;
	int n_h = ImgPr.ImagInfo.BmpInfo.height / mul;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Shrink(ImgPr.ImagInfo, out, mul);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool scaling(int mul, int mod)
{
	IMAGEINFOR out;
	int n_w, n_h;

	if (mod == ENLARGE)
	{
		n_w = ImgPr.ImagInfo.BmpInfo.width * mul;
		n_h = ImgPr.ImagInfo.BmpInfo.height * mul;
	}
	else if (mod == NARROW)
	{
		n_w = ImgPr.ImagInfo.BmpInfo.width / mul;
		n_h = ImgPr.ImagInfo.BmpInfo.height / mul;
	}

	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Scaling(ImgPr.ImagInfo, out, mul);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool rotation(float angle)
{
	IMAGEINFOR out;
	int n_w = ImgPr.ImagInfo.BmpInfo.width * 3;
	int n_h = ImgPr.ImagInfo.BmpInfo.height * 3;
	memcpy(&out.BmpFile, &ImgPr.ImagInfo.BmpFile, sizeof(out.BmpFile));
	memcpy(&out.BmpInfo, &ImgPr.ImagInfo.BmpInfo, sizeof(out.BmpInfo));
	out.BmpInfo.width = n_w;
	out.BmpInfo.height = n_h;
	out.BmpInfo.imagesize = (n_w) * (n_h) * 3;
	out.BmpInfo.ImgData = new BYTE[(n_w) * (n_h) * 3];
	bool flag = ImgPr.Rotation(ImgPr.ImagInfo, out, angle);
	memcpy(&ImgPr.ImagInfo.BmpFile, &out.BmpFile, sizeof(ImgPr.ImagInfo.BmpFile));
	memcpy(&ImgPr.ImagInfo.BmpInfo, &out.BmpInfo, sizeof(ImgPr.ImagInfo.BmpInfo));
	return flag;
}

bool gridlines(BYTE* color, int ncells)
{
	if (color == NULL) return false;
	return ImgPr.Gridlines(ImgPr.ImagInfo, color, ncells);
}

bool mlsr(int* srcP, int* dstP, int pn)
{
	return ImgPr.MLSR(ImgPr.ImagInfo, srcP, dstP, pn);
}

bool distortingMirrors(int x, int y, int k)
{
	return ImgPr.DistortingMirror(ImgPr.ImagInfo, x, y, k);
}

bool haar(int depth)
{
	return ImgPr.Haar(ImgPr.ImagInfo, depth);
}

bool kMeansCluster(int k)
{
	return ImgPr.KMeansCluster(ImgPr.ImagInfo, k);
}

bool segmentation()
{
	return ImgPr.Segmentation(ImgPr.ImagInfo);
}

bool matting()
{
	return ImgPr.Matting(ImgPr.ImagInfo, ImgPr.ImagInfo.BmpInfo.width, ImgPr.ImagInfo.BmpInfo.height);
}

bool oilpainting()
{
	return ImgPr.Oilpainting(ImgPr.ImagInfo);
}

bool atomization(int v)
{
	return ImgPr.Atomization(ImgPr.ImagInfo, v);
}

bool mosaic(int v)
{
	return ImgPr.Mosaic(ImgPr.ImagInfo, v);
}

bool sunlight(int x, int y, int r, float thd)
{
	return ImgPr.Sunlight(ImgPr.ImagInfo, x, y, r, thd);
}

bool noise(BYTE fn, double mv, double vv)
{
	return ImgPr.Noise(ImgPr.ImagInfo, fn, mv, vv);
}

bool floodfill(TPoint p, RGBColor fcolor, int thd)
{
	return ImgPr.FloodFill(ImgPr.ImagInfo, p, fcolor, thd);
}

bool fftImage(int mode)
{
	return ImgPr.FFTImage(ImgPr.ImagInfo, mode);
}

bool skinDetect()
{
	return ImgPr.SkinDetect(ImgPr.ImagInfo);
}

bool imageFeatureDetection(int mod)
{
	return ImgPr.ImageFeatureDetection(ImgPr.ImagInfo, mod);
}

bool cnn_()
{
	return ImgPr.CNN_();
}

bool dnn_()
{
	return ImgPr.DNN_();
}

void objectDetection_ISV(int mod)
{
	switch (mod)
	{
	case STREAMING_DETECTION:
		ImgPr.YoloV3();
		break;
	case IMAGE_DETECTION:
		ImgPr.YoloV3(ImgPr.ImagInfo);
		break;
	}
}

void objectDetection_V(const char* vstr, const char* sstr)
{
	ImgPr.YoloV3(vstr, sstr);
}

bool thresHold(int thd)
{
	return ImgPr.Threshold(ImgPr.ImagInfo, thd);
}