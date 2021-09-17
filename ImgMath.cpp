#include "pch.h"
#include "ImgMath.h"
#include <stdlib.h>
#include <malloc.h>

Imgmath::Imgmath()
{
	
}

Imgmath::~Imgmath()
{

}

bool Imgmath::Mean(unsigned char* inptr, double* bMean)
{
	int MeanNm = 0;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);
	for (int n = 0; n < N; n++)
		MeanNm += inptr[n];
	*bMean = MeanNm / N;
	return true;
}

bool Imgmath::MaxMin(unsigned char* inptr, double* Max, double* Min)
{
	double MaxNm, MinNm;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);
	MaxNm = MinNm = inptr[0];
	for (int n = 0; n < N; n++)
	{
		if (inptr[n] > MaxNm) MaxNm = inptr[n];
		else if (inptr[n] < MinNm) MinNm = inptr[n];
	}
	*Max = MaxNm;
	*Min = MinNm;
	return true;
}

bool Imgmath::STDEV(unsigned char* inptr, double* Stdev, double* Mean)
{
	double sum = 0;		//總和
	int ArrarNm = 0;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);	//計算元素個數N

	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		sum += inptr[ArrarNm];	//求總和
	*Mean = sum / N;				//計算平均值

	double temp = 0;
	double Spow = 0;
	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		Spow += (inptr[ArrarNm] - *Mean) * (inptr[ArrarNm] - *Mean);	//平方累加
	*Stdev = sqrt(Spow / N);
	return true;
}

bool Imgmath::F_STDEV(float* inptr, double* Stdev, double* Mean)
{
	double sum = 0;		//總和
	int ArrarNm = 0;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);	//計算元素個數N

	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		sum += inptr[ArrarNm];	//求總和
	*Mean = sum / N;				//計算平均值

	double temp = 0;
	double Spow = 0;
	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		Spow += (inptr[ArrarNm] - *Mean) * (inptr[ArrarNm] - *Mean);	//平方累加
	*Stdev = sqrt(Spow / N);
	return true;
}

double Imgmath::Logbase(double a,double base)
{
	return log(a) / log(base);
}