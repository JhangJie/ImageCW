#include "ImgMatting.h"
#include "ImgMath.h"
#include <windows.h>

#define	R_OK 4
#define	W_OK 2
#define	X_OK 1
#define	F_OK 0
#define BITWIDTH_8 8
#define BITWIDTH_16 16
#define BITWIDTH_32 32
#define M_PI 3.1415926
#define DQTALL 10
#define Round(x) ((int)x+0.5;))
#define MAX(a,b) ((a) > (b) ? (a) : (b) )
#define MIN(a,b) ((a) < (b) ? (a) : (b) )
#define CLIP3(x,a,b)MIN(MAX(a,x),b)
#define SWAP(x,y) { int t; t = x; x = y; y = t; }

const int DC = 0;
const int AC = 1;
typedef double BLOCK[8][8];

enum ImgBitAisle
{
	BIT8_1 = 8,
	BIT8_2 = 16,
	BIT8_3 = 24
};

enum TemplateMatchModes
{
	MN_COSINE_SIMILARITY = 0,
	MN_SQDIFF = 1,
	MN_SQDIFF_NORMED = 2,
	MN_CCORR = 3,
	MN_CCORR_NORMED = 4,
	MN_CCOEFF = 5,
	MN_CCOEFF_NORMED = 6
};

enum ImgFeatureDetection
{
	ImgFD_SIFT = 0,
	ImgFD_FAST_9 = 1,
	ImgFD_FAST_12 = 2,
	ImgFD_FAST_FAST_9 = 3,
	ImgFD_FAST_FAST_12 = 4
};

enum FFTMode
{
	FFTIMAGE = 0,
	FFTPHASEIMAGE = 1
};

enum ObjectdetectionMode
{
	IMAGE_DETECTION = 0,
	VIDEO_DETECTION = 1,
	STREAMING_DETECTION = 2
};

struct subVector
{
	unsigned char id;
	unsigned char width;
	unsigned char height;
	unsigned char quant;
};

struct acCode 
{
	unsigned char len;
	unsigned char zeros;
	int value;
};

typedef struct
{
	int x;
	int y;
}TPoint;

typedef struct
{
	BYTE rval;
	BYTE gval;
	BYTE bval;
}RGBColor;

typedef struct
{
	unsigned short int type;
	unsigned int size;							//File size in bytes,全部的檔案大小
	unsigned short int reserved1, reserved2;	//保留欄位
	unsigned int offset;
}BMPFILEHEADER;

typedef struct
{
	unsigned int size;				//Info Header size in bytes
	int width, height;				//Width and height of image
	unsigned short int planes;		//Number of colour planes
	unsigned short int bits;		//Bits per pixel
	unsigned int compression;		//Compression type
	unsigned int imagesize;			//Image size in bytes
	int xresolution, yresolution;	//Pixels per meter
	unsigned int ncolours;			//Number of colours
	unsigned int importantcolours;	//Important colours
	unsigned char* ImgData;
}BMPINFOHEADER;

typedef struct
{
	unsigned short app0length;	//APP0
	unsigned short horizontaldensity;
	unsigned short verticaldensity;
	unsigned char thumbnailhorizontalpixels;
	unsigned char thumbnailverticalpixels;

	unsigned short dqtlength;	//DQT
	unsigned int dqtnm;
	unsigned char** dqttable;
	float** dqttablerec;

	unsigned short dlength;		//SOF0
	unsigned char precision;
	unsigned short w, h;
	unsigned char colorcomponentnm;
	unsigned char** colorcomponentmsg;
	unsigned char** hvsamplingfactor;
	unsigned char hsmax;	//水平採樣
	unsigned char vsmax;	//垂直採樣

	unsigned short hufflength;	//DHT
	unsigned char httable;
	unsigned char* htlength;
	unsigned char** dchaffmansize;
	unsigned char** achaffmansize;
	unsigned int** dchaffmancoding;
	unsigned int** achaffmancoding;
	unsigned char** dchaffmanweight;
	unsigned char** achaffmanweight;

	unsigned short soslength;	//SOS
	unsigned char sosnmcomponents;
	unsigned char* soscolorcomponent;
	unsigned char* dchaffid;
	unsigned char* achaffid;

	unsigned int mcuw;			//MCU
	unsigned int mcuh;
	unsigned int stmcunm;
	unsigned int hlmcunm;

	char* mcu;
	unsigned char* Y;
	unsigned char* Cb;
	unsigned char* Cr;

}JPEGINFOHEADER;

typedef struct 
{
	BMPFILEHEADER BmpFile;
	BMPINFOHEADER BmpInfo;
	JPEGINFOHEADER JpegInfo;
}IMAGEINFOR;

class Sort
{
private:
	void TopDownSort(int* array, int* workArray, int length, int start, int count);
	void TopDownMerge(int* array, int* workArray, int length, int leftStart, int leftCount, int rightStart, int rightCount);
	void BottomUpMerge(int* array, int* workArray, int length, int leftStart, int leftCount, int rightStart, int rightCount);

public:
	bool InsertionSort(float* InPtrData);
	void TopDownSort(int* array, int length);
	void BottomUpSort(int* array, int length);

};

class ImageProcessing
{
private:

	//Load Iamge type
	bool Load_BMP(FILE* bFile);		//BMP
	bool Load_JPEG(FILE* bFile);	//JPEG

	//資料型態讀寫
	void Read_INT(FILE* fp, int* pRead);	//高低位元轉換
	void Read_UINT(FILE* fp, unsigned int* pRead);
	void Read_24BIT(FILE* fp, int* pRead);
	void Read_USH(FILE* fp, unsigned short* pRead);
	void Read_SH(FILE* fp, short* pRead);
	void Write_INT(FILE* fp, int nWrite);
	void Write_UINT(FILE* fp, unsigned int nWrite);
	void Write_24BIT(FILE* fp, unsigned int nWrite);
	void Write_USH(FILE* fp, unsigned short ushWrite);

	void Read_INT_N(FILE* fp, int* pRead);	//未高低位元轉換
	void Read_UINT_N(FILE* fp, unsigned int* pRead);
	void Read_24BIT_N(FILE* fp, int* pRead);
	void Read_USH_N(FILE* fp, unsigned short* pRead);
	void Read_SH_N(FILE* fp, short* pRead);
	void Write_INT_N(FILE* fp, int nWrite);
	void Write_UINT_N(FILE* fp, unsigned int nWrite);
	void Write_24BIT_N(FILE* fp, unsigned int nWrite);
	void Write_USH_N(FILE* fp, unsigned short ushWrite);

	bool GaussInverseMatrix(double* A, double* B, int nColumns);	//反矩陣
	void matrixMul(double* a, double* b, int m, int n, int k, double* c);

	//骨架化 Use
	int CountN(int* src);
	int CountT(int* src);

	//Image Understanding
	void DoG_SIFT(IMAGEINFOR in);
	void FAST(IMAGEINFOR in, int jd);
	void FAST_FAST(IMAGEINFOR in, int jd, int ctt);

	//JPEG
	void Show();
	void Quantify();
	void Zigzag();
	void Idct();
	void Decode();
	double Trans(int id, int h, int w);
	RGBColor** ToRGB();

	void ReadAPP(FILE* bFile);
	void ReadCOM(FILE* bFile);
	void ReadDQT(FILE* bFile);
	void ReadSOF(FILE* bFile);
	void ReadDHT(FILE* bFile);
	void ReadSOS(FILE* bFile);
	void ReadData(FILE* bFile);
	void ReadMCU(FILE* bFile);

	bool GetBit(FILE* bFile);
	int ReadDC(FILE* bFile, unsigned char number);
	acCode ReadAC(FILE* bFile, unsigned char number);
	unsigned char MatchHuff(FILE* bFile, unsigned char number, unsigned char ACorDC);

public:
	ImageProcessing();
	~ImageProcessing();

	Sort sort;
	IMAGEINFOR ImagInfo;
	BLOCK mcu[4][2][2];

	bool Load(FILE* bFile);
	bool Save(FILE* bFile);

	bool ReadImg(FILE* ptr, unsigned char* inptr);
	bool SaveImg(IMAGEINFOR in, const char* strpath, int w, int h);
	bool SaveImg_(char* inptr, const char* strpath, int w, int h);

	//Img運算
	bool GetValue(int x, int y, int* val);
	bool GetValue(int x, int y, float* val);
	bool PutValue(int x, int y, int *val);
	bool PutValue(int x, int y, float *val);
	bool ImgMean(double* dmean);
	bool ImgSTDEV(double* dstdev);
	bool ImgHsum(double* dhsum);
	bool ImgVsum(double* dvsum);
	bool ImgMaxMin(double* dmax, double* dmin);

	bool AddImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);
	bool SubtractImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);
	bool MultiplyImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);
	bool DivideImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);

	bool AndImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);
	bool OrImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);
	bool XorImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out);

	bool AddValue(IMAGEINFOR* in, int x, int y, int *val);
	bool AddValue(IMAGEINFOR* in, int x, int y, float *val);
	bool SubtractValue(IMAGEINFOR* in, int x, int y, int *val);
	bool SubtractValue(IMAGEINFOR* in, int x, int y, float *val);
	bool MultiplyValue(IMAGEINFOR* in, int x, int y, int *val);
	bool MultiplyValue(IMAGEINFOR* in, int x, int y, float *val);
	bool DivideValue(IMAGEINFOR* in, int x, int y, int *val);
	bool DivideValue(IMAGEINFOR* in, int x, int y, float *val);

	bool NotImage(IMAGEINFOR* in);
	bool LShiftImage(IMAGEINFOR* in, int Lsft);
	bool RShiftImage(IMAGEINFOR* in, int Rsft);

	bool AndValue(IMAGEINFOR* in, int x, int y, int *val);
	bool OrValue(IMAGEINFOR* in, int x, int y, int *val);
	bool XorValue(IMAGEINFOR* in, int x, int y, int *val);

	bool SetMask(int mask);

	//Color Mapping
	bool Reverse(IMAGEINFOR in);
	bool Grayscale(IMAGEINFOR in);
	bool RGrayscale(IMAGEINFOR in);		//待更新
	bool Binarization(IMAGEINFOR in);	//灰階二值化

	//直方圖
	bool Histogram(IMAGEINFOR in, double* histdata, int aisle, int shift);

	//Img應用
	bool Turnto(IMAGEINFOR in);			//上下轉換
	bool Conversion(IMAGEINFOR in);		//左右顛倒
	bool Stretch(IMAGEINFOR in, IMAGEINFOR out, int mul);		//拉寬
	bool Pullup(IMAGEINFOR in, IMAGEINFOR out, int mul);		//拉高
	bool Narrowing(IMAGEINFOR in, IMAGEINFOR out, int mul);	//縮寬
	bool Shrink(IMAGEINFOR in, IMAGEINFOR out, int mul);		//縮高
	bool Scaling(IMAGEINFOR in, IMAGEINFOR out, int mul);		//縮放

	bool Rotation(IMAGEINFOR in, IMAGEINFOR out, float theta);
	bool Gridlines(IMAGEINFOR in, unsigned char* color, int cells);

	bool f_MLSR(IMAGEINFOR in, int* srcP, int* dstP, int pNum);
	bool MLSR(IMAGEINFOR in, int* srcP, int* dstP, int pNum);

	bool DistortingMirror(IMAGEINFOR in, int x, int y, int k);

	//Filter
	bool S_Convolution(IMAGEINFOR in, float* kernel);	//一般速
	bool Convolution(IMAGEINFOR in, float* kernel);		//加速
	bool Sharpness(IMAGEINFOR in);
	bool Blur_BoxFilter(IMAGEINFOR in);
	bool Blur_GaussianFilter(IMAGEINFOR in);
	bool Blur_Motion(IMAGEINFOR in);
	bool MedianFilter(IMAGEINFOR in);		 //中值濾波器 ( 降噪 )
	bool FuzzyMedianFilter(IMAGEINFOR in);	 //中值濾波器 ( 降噪 ) 待修正
	bool PrewittFilter(IMAGEINFOR in);
	bool LoGFilter(IMAGEINFOR in, double sa, int krn);
	bool DoGFilter(IMAGEINFOR in, double sa, double sb, int krn);
	bool EdgeDetection_Sobelfilter(IMAGEINFOR in);
	bool EdgeDetection_LaplacianEdgeDetector(IMAGEINFOR in);
	bool Embossing(IMAGEINFOR in);
	bool KMeansCluster(IMAGEINFOR in, int k);
	bool Haar(IMAGEINFOR in, int depth);		//哈爾小波轉換 ( Harr )
	bool CAfilter(IMAGEINFOR in, int w, int h);	//圓形濾波器 ( 待修正 )

	bool Thinning_Zhang(IMAGEINFOR in);			//骨架化
	bool Segmentation(IMAGEINFOR in);
	bool Dilation(IMAGEINFOR in, int w, int h);	//膨脹
	bool Corrosion(IMAGEINFOR in, int w, int h);//侵蝕
	bool Matting(IMAGEINFOR in, int w, int h);	//待更新

	bool Blending(IMAGEINFOR in);		//合成	待新增
	bool Inpainting(IMAGEINFOR in);		//修補
	bool Completion(IMAGEINFOR in);		//補全
	bool Retargeting(IMAGEINFOR in);	//重新定位。縮小圖片長寬，裁剪多餘像素，但是保留景物原本形狀。

	//特徵擷取 , 判斷
	bool ImgRange(IMAGEINFOR in, IMAGEINFOR out, float* knl, int xst, int yst);
	bool TemplateMatching(IMAGEINFOR in, float* tel, int method);

	//Image Painting
	bool Oilpainting(IMAGEINFOR in);
	bool Atomization(IMAGEINFOR in, int v);
	bool Mosaic(IMAGEINFOR in, int v);
	bool Sunlight(IMAGEINFOR in, int x, int y, int r, float thd);
	bool Noise(IMAGEINFOR in, BYTE fn, double mv, double vv);
	bool FloodFill(IMAGEINFOR in, TPoint p, RGBColor fcolor, int thd);
	bool FFTImage(IMAGEINFOR in, int mode);

	//Image Understanding
	bool SkinDetect(IMAGEINFOR in);		//膚色檢測
	bool ImageFeatureDetection(IMAGEINFOR in, int mod);	//特徵偵測

	bool CNN_();
	bool DNN_();
	void YoloV3();										//攝像頭串流
	void YoloV3(IMAGEINFOR in);							//影像
	void YoloV3(const char* vstr, const char* ostr);	//影片

	bool ImageSemanticSegmentation();
	bool VideoObjectTracking();

	bool Threshold(IMAGEINFOR in, int thd);
};
