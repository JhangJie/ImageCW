#pragma once

#ifdef IMAGECW_DLL
#else
#define IMAGECW_DLL extern "C" __declspec(dllimport)	//指可以供外部使用
#endif

enum ScalingMod
{
	ENLARGE = 0,
	NARROW
};

IMAGECW_DLL bool loadImageFile(const char* pstr);
IMAGECW_DLL bool saveImageFile(const char* pstr);

IMAGECW_DLL bool getValue_INT(int x, int y, int* val);
IMAGECW_DLL bool getValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool putValue_INT(int x, int y, int* val);
IMAGECW_DLL bool putValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool imgMean(double* dmean);
IMAGECW_DLL bool imgSTDEV(double* dstdev);
IMAGECW_DLL bool imgHsum(double* dhsum);
IMAGECW_DLL bool imgVsum(double* dvsum);
IMAGECW_DLL bool imgMaxMin(double* dmax, double* dmin);
IMAGECW_DLL bool addImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool addImage_(const char* str2, const char* stro);
IMAGECW_DLL bool subtractImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool subtractImage_(const char* str2, const char* stro);
IMAGECW_DLL bool multiplyImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool multiplyImage_(const char* str2, const char* stro);
IMAGECW_DLL bool divideImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool divideImage_(const char* str2, const char* stro);
IMAGECW_DLL bool andImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool andImage_(const char* str2, const char* stro);
IMAGECW_DLL bool orImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool orImage_(const char* str2, const char* stro);
IMAGECW_DLL bool xorImage(const char* str1, const char* str2, const char* stro);
IMAGECW_DLL bool xorImage_(const char* str2, const char* stro);
IMAGECW_DLL bool addValue_INT(int x, int y, int* val);
IMAGECW_DLL bool addValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool subtractValue_INT(int x, int y, int* val);
IMAGECW_DLL bool subtractValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool multiplyValue_INT(int x, int y, int* val);
IMAGECW_DLL bool multiplyValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool divideValue_INT(int x, int y, int* val);
IMAGECW_DLL bool divideValue_FLO(int x, int y, float* val);
IMAGECW_DLL bool notImage();
IMAGECW_DLL bool lShiftImage(int Lsft);
IMAGECW_DLL bool rShiftImage(int Rsft);
IMAGECW_DLL bool andValue(int x, int y, int* val);
IMAGECW_DLL bool orValue(int x, int y, int* val);
IMAGECW_DLL bool xorValue(int x, int y, int* val);
IMAGECW_DLL bool setMask(int mask);

IMAGECW_DLL bool reverse();
IMAGECW_DLL bool grayscale();
IMAGECW_DLL bool binarization();
IMAGECW_DLL bool rGrayscale();
IMAGECW_DLL bool histogram(double* histdata, int aisle, int shift);
IMAGECW_DLL bool blurBoxFilter();
IMAGECW_DLL bool blurGaussianFilter();
IMAGECW_DLL bool blurMotion();
IMAGECW_DLL bool sharpNess();
IMAGECW_DLL bool edgeDetectionSobelfilter();
IMAGECW_DLL bool edgeDetectionLaplacianEdgeDetector();
IMAGECW_DLL bool embossing();
IMAGECW_DLL bool medianFilter();
IMAGECW_DLL bool fuzzyMedianFilter();
IMAGECW_DLL bool prewittFilter();
IMAGECW_DLL bool loGFilter();
IMAGECW_DLL bool doGFilter();
IMAGECW_DLL bool thresHold(int thd);

IMAGECW_DLL bool imgRange(float* kel, int x, int y);
IMAGECW_DLL bool templateMatching(float* kel, int mod);

IMAGECW_DLL bool turnto();
IMAGECW_DLL bool conversion();
IMAGECW_DLL bool stretch(int mul);
IMAGECW_DLL bool pullup(int mul);
IMAGECW_DLL bool narrowing(int mul);
IMAGECW_DLL bool shrink(int mul);
IMAGECW_DLL bool scaling(int mul, int mod);
IMAGECW_DLL bool rotation(float angle);

IMAGECW_DLL bool gridlines(BYTE* color, int ncells);
IMAGECW_DLL bool mlsr(int* srcP, int* dstP, int pn);
IMAGECW_DLL bool distortingMirrors(int x, int y, int k);
IMAGECW_DLL bool haar(int depth);
IMAGECW_DLL bool kMeansCluster(int k);
IMAGECW_DLL bool segmentation();
IMAGECW_DLL bool matting();
IMAGECW_DLL bool oilpainting();
IMAGECW_DLL bool atomization(int v);
IMAGECW_DLL bool mosaic(int v);
IMAGECW_DLL bool sunlight(int x, int y, int r, float thd);
IMAGECW_DLL bool noise(BYTE fn, double mv, double vv);
IMAGECW_DLL bool floodfill(TPoint p, RGBColor fcolor, int thd);
IMAGECW_DLL bool fftImage(int mode);
IMAGECW_DLL bool skinDetect();
IMAGECW_DLL bool imageFeatureDetection(int mod);

IMAGECW_DLL bool cnn_();
IMAGECW_DLL bool dnn_();
IMAGECW_DLL void objectDetection_ISV(int mod);
IMAGECW_DLL void objectDetection_V(const char* vstr, const char* sstr);