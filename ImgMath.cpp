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
	double sum = 0;		//�`�M
	int ArrarNm = 0;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);	//�p�⤸���Ӽ�N

	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		sum += inptr[ArrarNm];	//�D�`�M
	*Mean = sum / N;				//�p�⥭����

	double temp = 0;
	double Spow = 0;
	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		Spow += (inptr[ArrarNm] - *Mean) * (inptr[ArrarNm] - *Mean);	//����֥[
	*Stdev = sqrt(Spow / N);
	return true;
}

bool Imgmath::F_STDEV(float* inptr, double* Stdev, double* Mean)
{
	double sum = 0;		//�`�M
	int ArrarNm = 0;
	if (inptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);	//�p�⤸���Ӽ�N

	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		sum += inptr[ArrarNm];	//�D�`�M
	*Mean = sum / N;				//�p�⥭����

	double temp = 0;
	double Spow = 0;
	for (ArrarNm = 0; ArrarNm < N; ArrarNm++)
		Spow += (inptr[ArrarNm] - *Mean) * (inptr[ArrarNm] - *Mean);	//����֥[
	*Stdev = sqrt(Spow / N);
	return true;
}

double Imgmath::Logbase(double a,double base)
{
	return log(a) / log(base);
}