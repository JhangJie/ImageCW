#include "pch.h"
#include <random>
#include "ImgProcessing.h"
#include "FFT.h"
#include "DNN.h"
#include "CNN.h"
#include "ImgObjDetection.h"
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
#include <map>
using namespace std;

subVector subV[4];
double cos_cache[200];
int quantTable[4][128];
std::map<std::pair<unsigned char, unsigned int>, unsigned char> huffTable[2][2];

ImageProcessing::ImageProcessing()
{

}

ImageProcessing::~ImageProcessing()
{

}

//排序
bool Sort::InsertionSort(float* InPtrData)
{
	int n, m, tmp, N;
	if (InPtrData == NULL) return false;
	N = _msize(InPtrData) / sizeof(InPtrData[0]);
	for (n = 1; n < N; n++)
	{
		tmp = InPtrData[n];
		for (m = n; m > 0 && tmp < InPtrData[m - 1]; m--)
			InPtrData[m] = InPtrData[m - 1];
		InPtrData[m] = tmp;
	}
	return true;
}

void Sort::TopDownMerge(int* array, int* workArray, int length, int leftStart, int leftCount, int rightStart, int rightCount)
{
	int i = leftStart, j = rightStart, leftBound = leftStart + leftCount, rightBound = rightStart + rightCount, index = leftStart;
	while (i < leftBound || j < rightBound)
	{
		if (i < leftBound && j < rightBound)
		{
			if (array[j] < array[i])
				workArray[index] = array[j++];
			else
				workArray[index] = array[i++];
		}
		else if (i < leftBound)
			workArray[index] = array[i++];
		else
			workArray[index] = array[j++];
		++index;
	}
	for (i = leftStart; i < index; ++i)
		array[i] = workArray[i];
}

void Sort::TopDownSort(int* array, int* workArray, int length, int start, int count)
{
	if (count < 2)
		return;

	TopDownSort(array, workArray, length, start, count / 2);
	TopDownSort(array, workArray, length, start + count / 2, count - count / 2);
	TopDownMerge(array, workArray, length, start, count / 2, start + count / 2, count - count / 2);
}

void Sort::TopDownSort(int* array, int length)
{
	int* workArray = new int[length];
	TopDownSort(array, workArray, length, 0, length);
}

void Sort::BottomUpMerge(int* array, int* workArray, int length, int leftStart, int leftCount, int rightStart, int rightCount)
{
	int i = leftStart, j = rightStart, leftBound = leftStart + leftCount, rightBound = rightStart + rightCount, index = leftStart;
	while (i < leftBound || j < rightBound)
	{
		if (i < leftBound && j < rightBound)
		{
			if (array[j] < array[i])
				workArray[index] = array[j++];
			else
				workArray[index] = array[i++];
		}
		else if (i < leftBound)
			workArray[index] = array[i++];
		else
			workArray[index] = array[j++];
		++index;
	}
	for (i = leftStart; i < index; ++i)
		array[i] = workArray[i];
}

void Sort::BottomUpSort(int* array, int length)
{
	int* workArray = new int[length];

	for (int count = 1; count < length; count *= 2)
	{
		for (int leftStart = 0; leftStart < length; leftStart += 2 * count)
		{
			if (count > length - leftStart)
				break;
			BottomUpMerge(array, workArray, length, leftStart, count, leftStart + count, min(count, length - leftStart - count));
		}
	}
}

//資料型態讀寫
void ImageProcessing::Read_INT(FILE* fp, int* pRead)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	bByte4 = (BYTE)(dwBuf >> 24);
	*pRead = (bByte4 << 24) | (bByte3 << 16) | (bByte2 << 8) | bByte1;
}

void ImageProcessing::Read_UINT(FILE* fp, unsigned int* pRead)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	bByte4 = (BYTE)(dwBuf >> 24);
	*pRead = (bByte4 << 24) | (bByte3 << 16) | (bByte2 << 8) | bByte1;
}

void ImageProcessing::Read_24BIT(FILE* fp, int* pRead)
{
	BYTE bByte1, bByte2, bByte3;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	*pRead = (bByte3 << 16) | (bByte2 << 8) | bByte1;
}

void ImageProcessing::Read_USH(FILE* fp, unsigned short* pRead)
{
	BYTE bByte1, bByte2;
	DWORD wBuf;

	fread(&wBuf, sizeof(WORD), 1, fp);
	bByte1 = (BYTE)wBuf;
	bByte2 = (BYTE)(wBuf >> 8);
	*pRead = bByte1 << 8 | bByte2;
	*pRead = (bByte2 << 8) | bByte1;
}

void ImageProcessing::Read_SH(FILE* fp, short* pRead)
{
	BYTE bByte1, bByte2;
	DWORD wBuf;

	fread(&wBuf, sizeof(WORD), 1, fp);
	bByte1 = (BYTE)wBuf;
	bByte2 = (BYTE)(wBuf >> 8);
	*pRead = (bByte2 << 8) | bByte1;
}

void ImageProcessing::Write_INT(FILE* fp, int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 24;
	bByte2 = nWrite >> 16;
	bByte3 = nWrite >> 8;
	bByte4 = nWrite;
	dwBuf = (bByte1 << 24) | (bByte2 << 16) | (bByte3 << 8) | bByte4;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_UINT(FILE* fp, unsigned int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 24;
	bByte2 = nWrite >> 16;
	bByte3 = nWrite >> 8;
	bByte4 = nWrite;
	dwBuf = (bByte1 << 24) | (bByte2 << 16) | (bByte3 << 8) | bByte4;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_24BIT(FILE* fp, unsigned int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 16;
	bByte2 = nWrite >> 8;
	bByte3 = nWrite;
	dwBuf = (bByte1 << 16) | (bByte2 << 8) | bByte3;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_USH(FILE* fp, unsigned short ushWrite)
{
	BYTE bByte1, bByte2;
	WORD wBuf;

	bByte1 = (BYTE)(ushWrite >> 8);
	bByte2 = (BYTE)ushWrite;
	wBuf = (bByte1 << 8) | bByte2;
	fwrite(&wBuf, sizeof(WORD), 1, fp);
}

void ImageProcessing::Read_INT_N(FILE* fp, int* pRead)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	bByte4 = (BYTE)(dwBuf >> 24);
	*pRead = (bByte1 << 24) | (bByte2 << 16) | (bByte3 << 8)  | bByte4;
}

void ImageProcessing::Read_UINT_N(FILE* fp, unsigned int* pRead)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	bByte4 = (BYTE)(dwBuf >> 24);
	*pRead = (bByte1 << 24) | (bByte2 << 16) | (bByte3 << 8) | bByte4;
}

void ImageProcessing::Read_24BIT_N(FILE* fp, int* pRead)
{
	BYTE bByte1, bByte2, bByte3;
	DWORD dwBuf;

	fread(&dwBuf, sizeof(DWORD), 1, fp);
	bByte1 = (BYTE)dwBuf;
	bByte2 = (BYTE)(dwBuf >> 8);
	bByte3 = (BYTE)(dwBuf >> 16);
	*pRead = (bByte1 << 16) | (bByte2 << 8) | bByte3;
}

void ImageProcessing::Read_USH_N(FILE* fp, unsigned short* pRead)
{
	BYTE bByte1, bByte2;
	DWORD wBuf;

	fread(&wBuf, sizeof(WORD), 1, fp);
	bByte1 = (BYTE)wBuf;
	bByte2 = (BYTE)(wBuf >> 8);
	*pRead = bByte1 << 8 | bByte2;
}

void ImageProcessing::Read_SH_N(FILE* fp, short* pRead)
{
	BYTE bByte1, bByte2;
	DWORD wBuf;

	fread(&wBuf, sizeof(WORD), 1, fp);
	bByte1 = (BYTE)wBuf;
	bByte2 = (BYTE)(wBuf >> 8);
	*pRead = bByte1 << 8 | bByte2;
}

void ImageProcessing::Write_INT_N(FILE* fp, int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 24;
	bByte2 = nWrite >> 16;
	bByte3 = nWrite >> 8;
	bByte4 = nWrite;
	dwBuf = (bByte4 << 24) | (bByte3 << 16) | (bByte2 << 8) | bByte1;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_UINT_N(FILE* fp, unsigned int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 24;
	bByte2 = nWrite >> 16;
	bByte3 = nWrite >> 8;
	bByte4 = nWrite;
	dwBuf = (bByte4 << 24) | (bByte3 << 16) | (bByte2 << 8) | bByte1;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_24BIT_N(FILE* fp, unsigned int nWrite)
{
	BYTE bByte1, bByte2, bByte3, bByte4;
	DWORD dwBuf;

	bByte1 = nWrite >> 16;
	bByte2 = nWrite >> 8;
	bByte3 = nWrite;
	dwBuf = (bByte3 << 16) | (bByte2 << 8) | bByte1;
	fwrite(&dwBuf, sizeof(DWORD), 1, fp);
}

void ImageProcessing::Write_USH_N(FILE* fp, unsigned short ushWrite)
{
	BYTE bByte1, bByte2;
	WORD wBuf;

	bByte1 = (BYTE)(ushWrite >> 8);
	bByte2 = (BYTE)ushWrite;
	wBuf = (bByte2 << 8) | bByte1;
	fwrite(&wBuf, sizeof(WORD), 1, fp);
}

//進制轉換
string SixToBin(BYTE in)
{
	string ch;
	switch (in)
	{
	case 0x0:
		ch = "0000";
		break;
	case 0x1:
		ch = "0001";
		break;
	case 0x2:
		ch = "0010";
		break;
	case 0x3:
		ch = "0011";
		break;
	case 0x4:
		ch = "0100";
		break;
	case 0x5:
		ch = "0101";
		break;
	case 0x6:
		ch = "0110";
		break;
	case 0x7:
		ch = "0111";
		break;
	case 0x8:
		ch = "1000";
		break;
	case 0x9:
		ch = "1001";
		break;
	case 0xA:
		ch = "1010";
		break;
	case 0xB:
		ch = "1011";
		break;
	case 0xC:
		ch = "1100";
		break;
	case 0xD:
		ch = "1101";
		break;
	case 0xE:
		ch = "1110";
		break;
	case 0xF:
		ch = "1111";
		break;
	}
	return ch;
}

BYTE BinToSix(string in)
{
	BYTE ch = 0;
	if (in == "0000")      ch = 0x0;
	else if (in == "0001") ch = 0x1;
	else if (in == "0010") ch = 0x2;
	else if (in == "0011") ch = 0x3;
	else if (in == "0100") ch = 0x4;
	else if (in == "0101") ch = 0x5;
	else if (in == "0110") ch = 0x6;
	else if (in == "0111") ch = 0x7;
	else if (in == "1000") ch = 0x8;
	else if (in == "1001") ch = 0x9;
	else if (in == "1010") ch = 0xA;
	else if (in == "1011") ch = 0xB;
	else if (in == "1100") ch = 0xC;
	else if (in == "1101") ch = 0xD;
	else if (in == "1110") ch = 0xE;
	else if (in == "1111") ch = 0xF;
	return ch;
}

long BinToDec(char* in)
{
	int i;
	long nm = 0;
	char buff[256] = "0", cut[256] = "0";

	//直接轉 & 補0
	if (strlen(in) == 4)
	{
		nm = BinToSix(in);
	}
	else if (strlen(in) == 8)
	{
		for (i = 0; i < 4; i++)
			*(cut + i) = *(in + i);
		nm = BinToSix(cut) << 4;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(in + i);
		nm = nm | BinToSix(cut);
	}
	else if (strlen(in) == 12)
	{
		for (i = 0; i < 4; i++)
			*(cut + i) = *(in + i);
		nm = BinToSix(cut) << 8;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(in + i);
		nm = nm | BinToSix(cut) << 4;
		for (i = 8; i < 12; i++)
			*(cut + (i - 8)) = *(in + i);
		nm = nm | BinToSix(cut);
	}
	else if (strlen(in) == 16)
	{
		for (i = 0; i < 4; i++)
			*(cut + i) = *(in + i);
		nm = BinToSix(cut) << 12;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(in + i);
		nm = nm | BinToSix(cut) << 8;
		for (i = 8; i < 12; i++)
			*(cut + (i - 8)) = *(in + i);
		nm = nm | BinToSix(cut) << 4;
		for (i = 12; i < 16; i++)
			*(cut + (i - 12)) = *(in + i);
		nm = nm | BinToSix(cut);
	}
	else if (strlen(in) < 4)
	{
		for (i = 0; i < 4 - strlen(in); i++)
			strcpy(buff + i, "0");
		strcpy(buff + (4 - strlen(in)), in);
		nm = BinToSix(buff);
	}
	else if (strlen(in) > 4 && strlen(in) < 8)
	{
		for (i = 0; i < 8 - strlen(in); i++)
			strcpy(buff + i, "0");
		strcpy(buff + (8 - strlen(in)), in);
		for (i = 0; i < 4; i++)
			*(cut + i) = *(buff + i);
		nm = BinToSix(cut) << 4;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(buff + i);
		nm = nm | BinToSix(cut);
	}
	else if (strlen(in) > 8 && strlen(in) < 12)
	{
		for (i = 0; i < 12 - strlen(in); i++)
			strcpy(buff + i, "0");
		strcpy(buff + (12 - strlen(in)), in);
		for (i = 0; i < 4; i++)
			*(cut + i) = *(buff + i);
		nm = BinToSix(cut) << 8;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(buff + i);
		nm = nm | BinToSix(cut) << 4;
		for (i = 8; i < 12; i++)
			*(cut + (i - 8)) = *(buff + i);
		nm = nm | BinToSix(cut);
	}
	else if (strlen(in) > 12 && strlen(in) < 16)
	{
		for (i = 0; i < 16 - strlen(in); i++)
			strcpy(buff + i, "0");
		strcpy(buff + (16 - strlen(in)), in);
		for (i = 0; i < 4; i++)
			*(cut + i) = *(buff + i);
		nm = BinToSix(cut) << 12;
		for (i = 4; i < 8; i++)
			*(cut + (i - 4)) = *(buff + i);
		nm = nm | BinToSix(cut) << 8;
		for (i = 8; i < 12; i++)
			*(cut + (i - 8)) = *(buff + i);
		nm = nm | BinToSix(cut) << 4;
		for (i = 12; i < 16; i++)
			*(cut + (i - 12)) = *(buff + i);
		nm = nm | BinToSix(cut);
	}
	return nm;
}

char* Complement(char* in)
{
	int l = strlen(in);
	for (int i = 0; i < l; i++)
	{
		if (in[i] == '0')
			in[i] = '1';
		else if (in[i] == '1')
			in[i] = '0';
	}
	return in;
}

//JPEG
void init_cos_cache()
{
	for (int i = 0; i < 200; i++)
		cos_cache[i] = cos(i * M_PI / 16.0);
}

double cc(int i, int j)
{
	if (i == 0 && j == 0)
		return 1.0 / 2.0;
	else if (i == 0 || j == 0)
		return 1.0 / sqrt(2.0);
	else
		return 1.0;
}

double c(int i)
{
	static double x = 1.0 / sqrt(2.0);
	if (i == 0)
		return x;
	else
		return 1.0;
}

unsigned char chomp(double x)
{
	if (x > 255.0)
		return 255;
	else if (x < 0)
		return 0;
	else
		return (unsigned char)x;
}

double ImageProcessing::Trans(int id, int h, int w)
{
	int vh = h * subV[id].height / ImagInfo.JpegInfo.hsmax;
	int vw = w * subV[id].width / ImagInfo.JpegInfo.vsmax;
	return mcu[id][vh / 8][vw / 8][vh % 8][vw % 8];
}

void ImageProcessing::Show()
{
	FILE* ptr = fopen("C:\\Users\\CWChang\\Desktop\\test.log", "a+");
	fprintf(ptr, "*************** mcu show ***********************\n");
	for (int id = 1; id <= 3; id++) 
	{
		for (int h = 0; h < subV[id].height; h++)
		{
			for (int w = 0; w < subV[id].width; w++)
			{
				fprintf(ptr, "mcu id: %d, %d %d\n", id, h, w);
				for (int i = 0; i < 8; i++) 
				{
					for (int j = 0; j < 8; j++)
						fprintf(ptr, "%lf ", mcu[id][h][w][i][j]);
					fprintf(ptr, "\n");
				}
			}
		}
	}
	fclose(ptr);
};

void ImageProcessing::Quantify()
{
	for (int id = 1; id <= 3; id++)
		for (int h = 0; h < subV[id].height; h++)
			for (int w = 0; w < subV[id].width; w++)
				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
						mcu[id][h][w][i][j] *= quantTable[subV[id].quant][i * 8 + j];
};

void ImageProcessing::Zigzag()
{
	for (int id = 1; id <= 3; id++) 
	{
		for (int h = 0; h < subV[id].height; h++)
		{
			for (int w = 0; w < subV[id].width; w++)
			{
				int zz[8][8] = 
				{
						{ 0,  1,  5,  6, 14, 15, 27, 28},
						{ 2,  4,  7, 13, 16, 26, 29, 42},
						{ 3,  8, 12, 17, 25, 30, 41, 43},
						{ 9, 11, 18, 24, 31, 40, 44, 53},
						{10, 19, 23, 32, 39, 45, 52, 54},
						{20, 22, 33, 38, 46, 51, 55, 60},
						{21, 34, 37, 47, 50, 56, 59, 61},
						{35, 36, 48, 49, 57, 58, 62, 63}
				};

				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
						zz[i][j] = mcu[id][h][w][zz[i][j] / 8][zz[i][j] % 8];

				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
						mcu[id][h][w][i][j] = zz[i][j];
			}
		}
	}
};

void ImageProcessing::Idct()
{
	for (int id = 1; id <= 3; id++) 
	{
		for (int h = 0; h < subV[id].height; h++)
		{
			for (int w = 0; w < subV[id].width; w++)
			{
				double tmp[8][8] = { 0 };
				////照定義展開，效能低下
				//for (int i = 0; i < 8; i++) {
				//	for (int j = 0; j < 8; j++) {
				//		for (int x = 0; x < 8; x++) {
				//			for (int y = 0; y < 8; y++) {
				//				tmp[i][j] += (cc(x, y) * mcu.mcu[id][h][w][x][y] * cos((2*i+1)*M_PI/16.0*x) * cos((2*j+1)*M_PI/16.0*y));
				//			}
				//		}
				//		tmp[i][j] /= 4.0;
				//	}
				//}

				// 計算兩次一維idct去計算二維idct
				double s[8][8] = {};
				for (int j = 0; j < 8; j++) 
				{
					for (int x = 0; x < 8; x++) 
					{
						for (int y = 0; y < 8; y++) 
							s[j][x] += c(y) * mcu[id][h][w][x][y] * cos_cache[(j + j + 1) * y];
						s[j][x] = s[j][x] / 2.0;
					}
				}

				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8; j++)
					{
						for (int x = 0; x < 8; x++)
							tmp[i][j] += c(x) * s[j][x] * cos_cache[(i + i + 1) * x];
						tmp[i][j] = tmp[i][j] / 2.0;
					}
				}

				for (int i = 0; i < 8; i++)
					for (int j = 0; j < 8; j++)
						mcu[id][h][w][i][j] = tmp[i][j];
			}
		}
	}
}

void ImageProcessing::Decode()
{
	Quantify();
	Zigzag();
	Idct();
}

RGBColor** ImageProcessing::ToRGB()
{
	RGBColor** ret = (RGBColor**)malloc(sizeof(RGBColor**)* ImagInfo.JpegInfo.hsmax * 8);
	for (int i = 0; i < ImagInfo.JpegInfo.hsmax * 8; i++)
		ret[i] = (RGBColor*)malloc(sizeof(RGBColor*)* ImagInfo.JpegInfo.vsmax * 8);

	for (int i = 0; i < ImagInfo.JpegInfo.hsmax * 8; i++)
	{
		for (int j = 0; j < ImagInfo.JpegInfo.vsmax * 8; j++)
		{
			double Y  = Trans(1, i ,j);
			double Cb = Trans(2, i, j);
			double Cr = Trans(3, i, j);
			ret[i][j].rval = chomp(Y + 1.402 * Cr + 128);
			ret[i][j].gval = chomp(Y - 0.34414 * Cb - 0.71414 * Cr + 128);
			ret[i][j].bval = chomp(Y + 1.772 * Cb + 128);
		}
	}
	return ret;
}

bool ImageProcessing::GetBit(FILE* bFile)
{
	static unsigned char buf;
	static unsigned char count = 0;
	if (count == 0) 
	{
		fread(&buf, 1, 1, bFile);
		if (buf == 0xFF) 
		{
			unsigned char check = 0;
			fread(&check, 1, 1, bFile);
			if (check != 0x00) 
			{
				fprintf(stderr, "data 段有不是 0xFF00 的數據");
			}
		}
	}
	bool ret = buf & (1 << (7 - count));
	count = (count == 7 ? 0 : count + 1);
	return ret;
}

unsigned char ImageProcessing::MatchHuff(FILE* bFile, unsigned char number, unsigned char ACorDC)
{
	unsigned int len = 0;
	unsigned char codeLen;
	for (int count = 1; ; count++) 
	{
		len = len << 1;
		len += (unsigned int)GetBit(bFile);
		if (huffTable[ACorDC][number].find(std::make_pair(count, len)) != huffTable[ACorDC][number].end()) 
		{
			codeLen = huffTable[ACorDC][number][std::make_pair(count, len)];
			return codeLen;
		}
		if (count > 16) { fprintf(stderr, "key not found\n"); count = 1; len = 0; }
	}
}

int ImageProcessing::ReadDC(FILE* bFile, unsigned char number)
{
	unsigned char codeLen = MatchHuff(bFile, number, DC);
	if (codeLen == 0) { return 0; }
	unsigned char first = GetBit(bFile);
	int ret = 1;
	for (int i = 1; i < codeLen; i++) 
	{
		unsigned char b = GetBit(bFile);
		ret = ret << 1;
		ret += first ? b : !b;
	}
	ret = first ? ret : -ret;
	return ret;
}

acCode ImageProcessing::ReadAC(FILE* bFile, unsigned char number)
{
	unsigned char x = MatchHuff(bFile, number, AC);
	unsigned char zeros = x >> 4;
	unsigned char codeLen = x & 0x0F;

	if (x == 0) 
		return acCode{ 0,0,0 };
	else if (x == 0xF0) 
		return acCode{ 0, 16, 0 };

	unsigned  char first = GetBit(bFile);
	int code = 1;
	for (int i = 1; i < codeLen; i++) 
	{
		unsigned char b = GetBit(bFile);
		code = code << 1;
		code += first ? b : !b;
	}
	code = first ? code : -code;
	return acCode{ codeLen, zeros, code };
}

void ImageProcessing::ReadMCU(FILE* bFile)
{
	static int dc[4] = { 0, 0, 0, 0 };
	for (int i = 1; i <= 3; i++) 
	{
		for (int h = 0; h < subV[i].height; h++)
		{
			for (int w = 0; w < subV[i].width; w++)
			{
				dc[i] = ReadDC(bFile, i / 2) + dc[i];
				mcu[i][h][w][0][0] = dc[i];
				unsigned int count = 1;
				while (count < 64) 
				{
					acCode ac = ReadAC(bFile, i / 2);
					if (ac.len == 0 && ac.zeros == 16) 
					{
						for (int j = 0; j < ac.zeros; j++) 
						{
							mcu[i][h][w][count / 8][count % 8] = 0;
							count++;
						}
					}
					else if (ac.len == 0) 
					{
						break;
					}
					else 
					{
						for (int j = 0; j < ac.zeros; j++) 
						{
							mcu[i][h][w][count / 8][count % 8] = 0;
							count++;
						}
						mcu[i][h][w][count / 8][count % 8] = ac.value;
				 		count++;
					}
				}

				while (count < 64) 
				{
					mcu[i][h][w][count / 8][count % 8] = 0;
					count++;
				}
			}
		}
	}
}

void ImageProcessing::ReadData(FILE* bFile)
{
	printf("************************* test read data **********************************\n");
	int w = (ImagInfo.JpegInfo.w - 1) / (8 * ImagInfo.JpegInfo.hsmax) + 1;
	int h = (ImagInfo.JpegInfo.h - 1) / (8 * ImagInfo.JpegInfo.vsmax) + 1;
	printf("mcuw = %d mcuh = %d\n", w, h);

	//ptr = fopen("C:\\Users\\CWChang\\Desktop\\ToRGB.log", "a+");
	init_cos_cache();
	ImagInfo.BmpInfo.ImgData = new unsigned char[ImagInfo.JpegInfo.w * ImagInfo.JpegInfo.h * 3];
	memset(ImagInfo.BmpInfo.ImgData, 0, sizeof(unsigned char) * ImagInfo.JpegInfo.w * ImagInfo.JpegInfo.h * 3);
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			ReadMCU(bFile);
			Decode();
			RGBColor** b = ToRGB();
			for (int y = i * 8 * ImagInfo.JpegInfo.hsmax; y < (i + 1) * 8 * ImagInfo.JpegInfo.hsmax; y++)
			{
				for (int x = j * 8 * ImagInfo.JpegInfo.vsmax; x < (j + 1) * 8 * ImagInfo.JpegInfo.vsmax; x++)
				{
					int by = y - i * 8 * ImagInfo.JpegInfo.hsmax;
					int bx = x - j * 8 * ImagInfo.JpegInfo.vsmax;
					ImagInfo.BmpInfo.ImgData[(y * ImagInfo.JpegInfo.w * 3) + (3 * x)] = b[by][bx].bval;
					ImagInfo.BmpInfo.ImgData[(y * ImagInfo.JpegInfo.w * 3) + (3 * x) + 1] = b[by][bx].gval;
					ImagInfo.BmpInfo.ImgData[(y * ImagInfo.JpegInfo.w * 3) + (3 * x) + 2] = b[by][bx].rval;
					//fprintf(ptr, "[%d][%d]R = %d G = %d B = %d \n", y, x, b[by][bx].R, b[by][bx].G, b[by][bx].B);
				}
			}
			//Show();
		}
	}
	//fclose(ptr);

	ImagInfo.BmpFile.type = 0x4D42;
	ImagInfo.BmpFile.size = (ImagInfo.JpegInfo.w * ImagInfo.JpegInfo.h * 3) + 54;
	ImagInfo.BmpFile.reserved1 = 0;
	ImagInfo.BmpFile.reserved2 = 0;
	ImagInfo.BmpFile.offset = 54;
	ImagInfo.BmpInfo.size   = 40;
	ImagInfo.BmpInfo.width  = ImagInfo.JpegInfo.w;
	ImagInfo.BmpInfo.height = ImagInfo.JpegInfo.h;
	ImagInfo.BmpInfo.planes = 1;
	ImagInfo.BmpInfo.bits   = 24;
	ImagInfo.BmpInfo.compression = 0;
	ImagInfo.BmpInfo.imagesize = ImagInfo.JpegInfo.w * ImagInfo.JpegInfo.h * 3;
	ImagInfo.BmpInfo.xresolution = 0;
	ImagInfo.BmpInfo.yresolution = 0;
	ImagInfo.BmpInfo.ncolours    = 0;
	ImagInfo.BmpInfo.importantcolours = 0;

	Turnto(ImagInfo);
	Conversion(ImagInfo);
}

void ImageProcessing::ReadSOS(FILE* bFile)
{
	int next = -1;
	printf("SOS - Start of Scan 開始掃描\n");

	Read_USH_N(bFile, &ImagInfo.JpegInfo.soslength);
	printf("soslength = %d\n", ImagInfo.JpegInfo.soslength);

	fread(&ImagInfo.JpegInfo.sosnmcomponents, sizeof(BYTE), 1, bFile);
	printf("sosnmcomponents = %d\n", ImagInfo.JpegInfo.sosnmcomponents);

	ImagInfo.JpegInfo.dchaffid = new unsigned char[ImagInfo.JpegInfo.sosnmcomponents];
	ImagInfo.JpegInfo.achaffid = new unsigned char[ImagInfo.JpegInfo.sosnmcomponents];
	ImagInfo.JpegInfo.soscolorcomponent = new unsigned char[ImagInfo.JpegInfo.sosnmcomponents * 2];
	for (int i = 0; i < ImagInfo.JpegInfo.sosnmcomponents * 2; i++)
	{
		fread(&ImagInfo.JpegInfo.soscolorcomponent[i], sizeof(BYTE), 1, bFile);
		if ((i % 2) != 0)
		{
			next++;
			ImagInfo.JpegInfo.dchaffid[next] = ImagInfo.JpegInfo.soscolorcomponent[i] >> 4;
			ImagInfo.JpegInfo.achaffid[next] = ImagInfo.JpegInfo.soscolorcomponent[i] & 0x0F;
			printf("dchaffid = %X achaffid = %X\n", ImagInfo.JpegInfo.dchaffid[next], ImagInfo.JpegInfo.achaffid[next]);
		}
		//printf("soscolorcomponent = %X\n", ImagInfo.JpegInfo.soscolorcomponent[i]);
	}
	fseek(bFile, 3, SEEK_CUR);
}

std::pair<unsigned char, unsigned int>* createHuffCode(unsigned char* a, unsigned int number)
{
	int si = sizeof(std::pair<unsigned char, unsigned int>);
	auto ret = (std::pair<unsigned char, unsigned int>*)malloc(si * number);
	int code = 0;
	int count = 0;
	for (int i = 0; i < 16; i++)
	{
		for (int j = 0; j < a[i]; j++)
		{
			ret[count++] = std::make_pair(i + 1, code);
			code += 1;
		}
		code = code << 1;
	}
	return ret;
}

void ImageProcessing::ReadDHT(FILE* bFile)
{
	BYTE wBuf;
	int number = 0;
	BYTE* buf  = new BYTE[16];
	memset(buf, 0, sizeof(BYTE) * 16);

	printf("DHT - Huffman 表\n");

	Read_USH_N(bFile, &ImagInfo.JpegInfo.hufflength);
	printf("hufflength = %d\n", ImagInfo.JpegInfo.hufflength);

	fread(&ImagInfo.JpegInfo.httable, sizeof(BYTE), 1, bFile);
	printf("httable = %X\n", ImagInfo.JpegInfo.httable);

	for (int i = 0; i < 16; i++)
	{
		fread(&wBuf, sizeof(BYTE), 1, bFile);
		number += wBuf;
		buf[i] = wBuf;
	}

	unsigned char DCorAC = ImagInfo.JpegInfo.httable >> 4;
	unsigned char id     = ImagInfo.JpegInfo.httable & 0x0F;
	printf(DCorAC == 0 ? "DC\n" : "AC\n");
	printf("ID: %d\n", id);

	auto huffCode = createHuffCode(buf, number);
	for (int i = 0; i < number; i++)
		fread(&huffTable[DCorAC][id][huffCode[i]], sizeof(BYTE), 1, bFile);
	free(huffCode);
}

void ImageProcessing::ReadSOF(FILE* bFile)
{
	unsigned char v[3];
	printf("SOF0 - Start of Frame 非離散餘弦變化\n");

	Read_USH_N(bFile, &ImagInfo.JpegInfo.dlength);
	printf("dlength = %d\n", ImagInfo.JpegInfo.dlength);

	fread(&ImagInfo.JpegInfo.precision, sizeof(BYTE), 1, bFile);
	printf("precision = %d\n", ImagInfo.JpegInfo.precision);

	Read_USH_N(bFile, &ImagInfo.JpegInfo.h);
	printf("h = %d\n", ImagInfo.JpegInfo.h);

	Read_USH_N(bFile, &ImagInfo.JpegInfo.w);
	printf("w = %d\n", ImagInfo.JpegInfo.w);

	fread(&ImagInfo.JpegInfo.colorcomponentnm, sizeof(BYTE), 1, bFile);
	printf("colorcomponent = %d\n", ImagInfo.JpegInfo.colorcomponentnm);

	for (int i = 0; i < 3; i++)
	{
		fread(v, 1, 3, bFile);
		printf("顏色分量ID：%d\n", v[0]);
		printf("水平採樣因子：%d\n", v[1] >> 4);
		printf("垂直採樣因子：%d\n", v[1] & 0x0F);
		printf("量化表ID：%d\n", v[2]);
		subV[v[0]].id = v[0];
		subV[v[0]].width = v[1] >> 4;
		subV[v[0]].height = v[1] & 0x0F;
		subV[v[0]].quant = v[2];
		ImagInfo.JpegInfo.vsmax = (ImagInfo.JpegInfo.vsmax > subV[v[0]].height ? ImagInfo.JpegInfo.vsmax : subV[v[0]].height);
		ImagInfo.JpegInfo.hsmax = (ImagInfo.JpegInfo.hsmax > subV[v[0]].width ? ImagInfo.JpegInfo.hsmax : subV[v[0]].width);
	}
	printf("水平Max = %X 垂直Max = %X\n", ImagInfo.JpegInfo.hsmax, ImagInfo.JpegInfo.vsmax);
}

void ImageProcessing::ReadDQT(FILE* bFile)
{
	int precision, id;
	printf("DQT - Define Quantization Table 定義表\n");

	Read_USH_N(bFile, &ImagInfo.JpegInfo.dqtlength);
	printf("DQTlength = %d\n", ImagInfo.JpegInfo.dqtlength);

	fread(&ImagInfo.JpegInfo.dqtnm, sizeof(BYTE), 1, bFile);
	printf("DQT table Nmber = %d\n", ImagInfo.JpegInfo.dqtnm);

	precision = ImagInfo.JpegInfo.dqtnm >> 4 == 0 ? 8 : 16;
	printf("精度：%d\n", precision);
	precision /= 8;
	id = ImagInfo.JpegInfo.dqtnm & 0x0F;
	printf("量化表ID: %d\n", id);

	for (int i = 0; i < 64; i++)
		fread(&quantTable[id][i], sizeof(BYTE), 1, bFile);

	for (int i = 0; i < 64; i++)
	{
		if (i % 8 == 0)
			printf("\n");
		printf("%2d ", quantTable[id][i]);
	}
	printf("\n");
}

void ImageProcessing::ReadCOM(FILE* bFile)
{
	printf("COM - 注譯\n");

	unsigned char c;
	unsigned short len;
	Read_USH_N(bFile, &len);
	printf("COMlength = %d\n", len);

	for (int i = 0; i < len - 2; i++) 
	{
		fread(&c, 1, 1, bFile);
		printf("%c", c);
	}
	printf("\n");
}

void ImageProcessing::ReadAPP(FILE* bFile)
{
	printf("APP0 - Application 應用程序保留標記\n");

	Read_USH_N(bFile, &ImagInfo.JpegInfo.app0length);
	printf("app0length = %d\n", ImagInfo.JpegInfo.app0length);

	fseek(bFile, 8, SEEK_CUR);

	Read_USH_N(bFile, &ImagInfo.JpegInfo.horizontaldensity);
	printf("horizontaldensity = %d\n", ImagInfo.JpegInfo.horizontaldensity);

	Read_USH_N(bFile, &ImagInfo.JpegInfo.verticaldensity);
	printf("verticaldensity = %d\n", ImagInfo.JpegInfo.verticaldensity);

	fread(&ImagInfo.JpegInfo.thumbnailhorizontalpixels, sizeof(BYTE), 1, bFile);
	printf("thumbnailhorizontalpixels = %d\n", ImagInfo.JpegInfo.thumbnailhorizontalpixels);

	fread(&ImagInfo.JpegInfo.thumbnailverticalpixels, sizeof(BYTE), 1, bFile);
	printf("thumbnailverticalpixels = %d\n", ImagInfo.JpegInfo.thumbnailverticalpixels);
}

bool ImageProcessing::Load_JPEG(FILE* bFile)	//JPEG
{
	if (bFile == NULL) return false;
	BYTE bB1, bB2;
	fread(&bB1, sizeof(BYTE), 1, bFile);
	fread(&bB2, sizeof(BYTE), 1, bFile);

	while (bB2 != 0xD9)
	{
		fread(&bB1, sizeof(BYTE), 1, bFile);
		if (bB1 == 0xFF)
		{
			fread(&bB2, sizeof(BYTE), 1, bFile);
			switch (bB2)
			{
			case 0xE0:
			case 0xE1:
			case 0xE2:
			case 0xE3:
			case 0xE4:
			case 0xE5:
			case 0xE6:
			case 0xE7:
			case 0xE8:
			case 0xE9:
			case 0xEA:
			case 0xEB:
			case 0xEC:
			case 0xED:
			case 0xEE:
			case 0xEF:
				ReadAPP(bFile);
				break;
			case 0xFE:
				ReadCOM(bFile);
				break;
			case 0xDB:
				ReadDQT(bFile);
				break;
			case 0xC0:
				ReadSOF(bFile);
				break;
			case 0xC4:
				ReadDHT(bFile);
				break;
			case 0xDA:
				ReadSOS(bFile);
				ReadData(bFile);
				break;
			case 0xDD:
				printf("DRI - 定義重新開始間隔\n");
				break;
			case 0xD9:
				printf("EOI - END\n");
				break;
			case 0x01:
			case 0xD0:
			case 0xD1:
			case 0xD2:
			case 0xD3:
			case 0xD4:
			case 0xD5:
			case 0xD6:
			case 0xD7:
				break;
			default:
				break;
			}
		}
	}
	return true;
}

bool ImageProcessing::Load_BMP(FILE* bFile)	//BMP
{
	Read_UINT(bFile, &ImagInfo.BmpFile.size);
	Read_USH(bFile, &ImagInfo.BmpFile.reserved1);
	Read_USH(bFile, &ImagInfo.BmpFile.reserved2);
	Read_UINT(bFile, &ImagInfo.BmpFile.offset);
	Read_UINT(bFile, &ImagInfo.BmpInfo.size);
	Read_INT(bFile, &ImagInfo.BmpInfo.width);
	Read_INT(bFile, &ImagInfo.BmpInfo.height);
	Read_USH(bFile, &ImagInfo.BmpInfo.planes);
	Read_USH(bFile, &ImagInfo.BmpInfo.bits);
	Read_UINT(bFile, &ImagInfo.BmpInfo.compression);
	Read_UINT(bFile, &ImagInfo.BmpInfo.imagesize);
	Read_INT(bFile, &ImagInfo.BmpInfo.xresolution);
	Read_INT(bFile, &ImagInfo.BmpInfo.yresolution);
	Read_UINT(bFile, &ImagInfo.BmpInfo.ncolours);
	Read_UINT(bFile, &ImagInfo.BmpInfo.importantcolours);

	printf("width = %d\n", ImagInfo.BmpInfo.width);
	printf("height = %d\n", ImagInfo.BmpInfo.height);

	//Image Data
	ImagInfo.BmpInfo.ImgData = new BYTE[ImagInfo.BmpInfo.imagesize];
	fread(ImagInfo.BmpInfo.ImgData, sizeof(BYTE), ImagInfo.BmpInfo.imagesize, bFile);
	return true;
}

//讀寫檔
bool ImageProcessing::Load(FILE* bFile)
{
	if (bFile == NULL) return false;
	unsigned short type;
	Read_USH(bFile, &type);
	printf("type = %X\n", type);
	switch (type)
	{
	case 0x4D42:
		printf("Load_BMP()\n");
		ImagInfo.BmpFile.type = type;
		Load_BMP(bFile);
		break;
	case 0xD8FF:
		printf("Load_JPEG()\n");
		Load_JPEG(bFile);
		break;
	}
	return true;
}

bool ImageProcessing::Save(FILE* bFile)
{
	if (bFile == NULL) return false;

	//Header Info
	Write_USH(bFile, ImagInfo.BmpFile.type);
	Write_UINT(bFile, ImagInfo.BmpFile.size);
	Write_USH(bFile, ImagInfo.BmpFile.reserved1);
	Write_USH(bFile, ImagInfo.BmpFile.reserved2);
	Write_UINT(bFile, ImagInfo.BmpFile.offset);
	Write_UINT(bFile, ImagInfo.BmpInfo.size);
	Write_INT(bFile, ImagInfo.BmpInfo.width);
	Write_INT(bFile, ImagInfo.BmpInfo.height);
	Write_USH(bFile, ImagInfo.BmpInfo.planes);
	Write_USH(bFile, ImagInfo.BmpInfo.bits);
	Write_UINT(bFile, ImagInfo.BmpInfo.compression);
	Write_UINT(bFile, ImagInfo.BmpInfo.imagesize);
	Write_INT(bFile, ImagInfo.BmpInfo.xresolution);
	Write_INT(bFile, ImagInfo.BmpInfo.yresolution);
	Write_UINT(bFile, ImagInfo.BmpInfo.ncolours);
	Write_UINT(bFile, ImagInfo.BmpInfo.importantcolours);
	
	//Image Data
	fwrite(ImagInfo.BmpInfo.ImgData, sizeof(BYTE), ImagInfo.BmpInfo.imagesize, bFile);
	return true;
}

bool ImageProcessing::ReadImg(FILE* ptr, unsigned char* inptr)
{
	if (ptr == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);

	BMPFILEHEADER BmpF;
	BMPINFOHEADER BmpI;

	Read_USH(ptr, &BmpF.type);
	Read_UINT(ptr, &BmpF.size);
	Read_USH(ptr, &BmpF.reserved1);
	Read_USH(ptr, &BmpF.reserved2);
	Read_UINT(ptr, &BmpF.offset);
	Read_UINT(ptr, &BmpI.size);
	Read_INT(ptr, &BmpI.width);
	Read_INT(ptr, &BmpI.height);
	Read_USH(ptr, &BmpI.planes);
	Read_USH(ptr, &BmpI.bits);
	Read_UINT(ptr, &BmpI.compression);
	Read_UINT(ptr, &BmpI.imagesize);
	Read_INT(ptr, &BmpI.xresolution);
	Read_INT(ptr, &BmpI.yresolution);
	Read_UINT(ptr, &BmpI.ncolours);
	Read_UINT(ptr, &BmpI.importantcolours);
	printf("BmpI.width = %d\n", BmpI.imagesize);
	//inptr = new BYTE[BmpI.imagesize];
	fread(inptr, sizeof(BYTE), BmpI.imagesize, ptr);
	return true;
}

bool ImageProcessing::SaveImg(IMAGEINFOR in, const char* strpath, int w, int h)
{
	if (strpath == NULL) return false;
	FILE* ptr = NULL;
	ptr = fopen(strpath, "wb");
	Write_USH(ptr, in.BmpFile.type);
	Write_UINT(ptr, in.BmpInfo.imagesize + 54);
	Write_USH(ptr, in.BmpFile.reserved1);
	Write_USH(ptr, in.BmpFile.reserved2);
	Write_UINT(ptr, in.BmpFile.offset);
	Write_UINT(ptr, in.BmpInfo.size);
	Write_INT(ptr, w);
	Write_INT(ptr, h);
	Write_USH(ptr, in.BmpInfo.planes);
	Write_USH(ptr, in.BmpInfo.bits);
	Write_UINT(ptr, in.BmpInfo.compression);
	Write_UINT(ptr, in.BmpInfo.imagesize);
	Write_INT(ptr, in.BmpInfo.xresolution);
	Write_INT(ptr, in.BmpInfo.yresolution);
	Write_UINT(ptr, in.BmpInfo.ncolours);
	Write_UINT(ptr, in.BmpInfo.importantcolours);
	fwrite(in.BmpInfo.ImgData, sizeof(BYTE), in.BmpInfo.imagesize, ptr);
	fclose(ptr);
	return true;
}

bool ImageProcessing::SaveImg_(char* inptr, const char* strpath, int w, int h)
{
	if (strpath == NULL) return false;
	int N = _msize(inptr) / sizeof(inptr[0]);

	FILE* ptr = NULL;
	ptr = fopen(strpath, "wb");
	Write_USH(ptr, ImagInfo.BmpFile.type);
	Write_UINT(ptr, N + 54);
	Write_USH(ptr, ImagInfo.BmpFile.reserved1);
	Write_USH(ptr, ImagInfo.BmpFile.reserved2);
	Write_UINT(ptr, ImagInfo.BmpFile.offset);
	Write_UINT(ptr, ImagInfo.BmpInfo.size);
	Write_INT(ptr, w);
	Write_INT(ptr, h);
	Write_USH(ptr, ImagInfo.BmpInfo.planes);
	Write_USH(ptr, ImagInfo.BmpInfo.bits);
	Write_UINT(ptr, ImagInfo.BmpInfo.compression);
	Write_UINT(ptr, N);
	Write_INT(ptr, ImagInfo.BmpInfo.xresolution);
	Write_INT(ptr, ImagInfo.BmpInfo.yresolution);
	Write_UINT(ptr, ImagInfo.BmpInfo.ncolours);
	Write_UINT(ptr, ImagInfo.BmpInfo.importantcolours);
	fwrite(inptr, sizeof(char), N, ptr);
	fclose(ptr);
	return true;
}

//Img運算
bool ImageProcessing::GetValue(int x, int y, int* val)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	if (ImagInfo.BmpInfo.bits == BIT8_1)
	{
		*val = ImagInfo.BmpInfo.ImgData[y* ImagInfo.BmpInfo.width + x];
	}
	else if (ImagInfo.BmpInfo.bits == BIT8_3)
	{
		val[0] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)];
		val[1] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x) + 1];
		val[2] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x) + 2];
	}
	return true;
}

bool ImageProcessing::GetValue(int x, int y, float* val)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	if (ImagInfo.BmpInfo.bits == BIT8_1)
	{
		*val = ImagInfo.BmpInfo.ImgData[y* ImagInfo.BmpInfo.width + x];
	}
	else if (ImagInfo.BmpInfo.bits == BIT8_3)
	{
		val[0] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)];
		val[1] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x) + 1];
		val[2] = ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x) + 2];
	}
	return true;
}

bool ImageProcessing::PutValue(int x, int y, int* val)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	if (ImagInfo.BmpInfo.bits == BIT8_1)
	{
		ImagInfo.BmpInfo.ImgData[y * ImagInfo.BmpInfo.width + x] = *val;
	}
	else if (ImagInfo.BmpInfo.bits == BIT8_3)
	{
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[0];
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[1];
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[2];
	}
	return true;
}

bool ImageProcessing::PutValue(int x, int y, float *val)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	if (ImagInfo.BmpInfo.bits == BIT8_1)
	{
		ImagInfo.BmpInfo.ImgData[y * ImagInfo.BmpInfo.width + x] = *val;
	}
	else if (ImagInfo.BmpInfo.bits == BIT8_3)
	{
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[0];
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[1];
		ImagInfo.BmpInfo.ImgData[(y * ImagInfo.BmpInfo.width * 3) + (3 * x)] = val[2];
	}
	return true;
}

bool ImageProcessing::ImgMean(double* dmean)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	Imgmath math;
	math.Mean(ImagInfo.BmpInfo.ImgData, dmean);
	return true;
}

bool ImageProcessing::ImgSTDEV(double* dstdev)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	Imgmath math;
	double mean;
	math.Mean(ImagInfo.BmpInfo.ImgData, &mean);
	math.STDEV(ImagInfo.BmpInfo.ImgData, dstdev, &mean);
	return true;
}

bool ImageProcessing::ImgHsum(double* dhsum)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;


	return true;
}

bool ImageProcessing::ImgVsum(double* dvsum)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;


	return true;
}

bool ImageProcessing::ImgMaxMin(double* dmax, double* dmin)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	Imgmath math;
	math.MaxMin(ImagInfo.BmpInfo.ImgData, dmax, dmin);
	return true;
}

bool ImageProcessing::AddImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] + in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] + in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] + in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] + in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::SubtractImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] - in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] - in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] - in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] - in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::MultiplyImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] * in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] * in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] * in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] * in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::DivideImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] / in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				BYTE d1 = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				BYTE d2 = in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				if (d1 == 0 || d2 == 0)
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = 0;
				else
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] / in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];

				d1 = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				d2 = in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				if (d1 == 0 || d2 == 0)
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = 0;
				else
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] / in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];

				d1 = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
				d2 = in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
				if (d1 == 0 || d2 == 0)
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = 0;
				else
					out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] / in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::AndImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] & in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] & in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] & in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] & in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::OrImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] | in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] | in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] | in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] | in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::XorImage(IMAGEINFOR in1, IMAGEINFOR in2, IMAGEINFOR* out)
{
	if (in1.BmpInfo.ImgData == NULL || in2.BmpInfo.ImgData == NULL) return false;
	if (in1.BmpInfo.imagesize != in2.BmpInfo.imagesize) return false;
	if (in1.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in1.BmpInfo.imagesize; n++)
			out->BmpInfo.ImgData[n] = in1.BmpInfo.ImgData[n] ^ in2.BmpInfo.ImgData[n];
	}
	else if (in1.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in1.BmpInfo.height; n++)
		{
			for (int m = 0; m < in1.BmpInfo.width; m++)
			{
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)] ^ in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m)];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1] ^ in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 1];
				out->BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] = in1.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2] ^ in2.BmpInfo.ImgData[(n * in1.BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::AddValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] += *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] += val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] += val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] += val[2];
	}
	return true;
}

bool ImageProcessing::AddValue(IMAGEINFOR* in, int x, int y, float *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] += *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] += val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] += val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] += val[2];
	}
	return true;
}

bool ImageProcessing::SubtractValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] -= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] -= val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] -= val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] -= val[2];
	}
	return true;
}

bool ImageProcessing::SubtractValue(IMAGEINFOR* in, int x, int y, float *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] -= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] -= val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] -= val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] -= val[2];
	}
	return true;
}

bool ImageProcessing::MultiplyValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] *= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] *= val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] *= val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] *= val[2];
	}
	return true;
}

bool ImageProcessing::MultiplyValue(IMAGEINFOR* in, int x, int y, float *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] *= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] *= val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] *= val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] *= val[2];
	}
	return true;
}

bool ImageProcessing::DivideValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] /= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		BYTE d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)];
		BYTE d2 = val[0];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] /= val[0];

		d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1];
		d2 = val[1];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] /= val[1];

		d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2];
		d2 = val[2];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] /= val[2];
	}
	return true;
}

bool ImageProcessing::DivideValue(IMAGEINFOR* in, int x, int y, float *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] /= *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		BYTE d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)];
		BYTE d2 = val[0];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] /= val[0];

		d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1];
		d2 = val[1];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] /= val[1];

		d1 = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2];
		d2 = val[2];
		if (d1 == 0 || d2 == 0)
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] = 0;
		else
			in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] /= val[2];
	}
	return true;
}

bool ImageProcessing::NotImage(IMAGEINFOR* in)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in->BmpInfo.imagesize; n++)
			in->BmpInfo.ImgData[n] = !in->BmpInfo.ImgData[n];
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in->BmpInfo.height; n++)
		{
			for (int m = 0; m < in->BmpInfo.width; m++)
			{
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)] = !in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)];
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1] = !in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1];
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2] = !in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2];
			}
		}
	}
	return true;
}

bool ImageProcessing::LShiftImage(IMAGEINFOR* in, int Lsft)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in->BmpInfo.imagesize; n++)
			in->BmpInfo.ImgData[n] = in->BmpInfo.ImgData[n] << Lsft;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in->BmpInfo.height; n++)
		{
			for (int m = 0; m < in->BmpInfo.width; m++)
			{
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)] << Lsft;
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1] << Lsft;
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2] << Lsft;
			}
		}
	}
	return true;
}

bool ImageProcessing::RShiftImage(IMAGEINFOR* in, int Rsft)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < in->BmpInfo.imagesize; n++)
			in->BmpInfo.ImgData[n] = in->BmpInfo.ImgData[n] >> Rsft;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < in->BmpInfo.height; n++)
		{
			for (int m = 0; m < in->BmpInfo.width; m++)
			{
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m)] >> Rsft;
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 1] >> Rsft;
				in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2] = in->BmpInfo.ImgData[(n * in->BmpInfo.width * 3) + (3 * m) + 2] >> Rsft;
			}
		}
	}
	return true;
}

bool ImageProcessing::AndValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] = in->BmpInfo.ImgData[y * in->BmpInfo.width + x] & *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] & val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] & val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] & val[2];
	}
	return true;
}

bool ImageProcessing::OrValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] = in->BmpInfo.ImgData[y * in->BmpInfo.width + x] | *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] | val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] | val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] | val[2];
	}
	return true;
}

bool ImageProcessing::XorValue(IMAGEINFOR* in, int x, int y, int *val)
{
	if (in->BmpInfo.ImgData == NULL) return false;
	if (in->BmpInfo.bits == BIT8_1)
	{
		in->BmpInfo.ImgData[y * in->BmpInfo.width + x] = in->BmpInfo.ImgData[y * in->BmpInfo.width + x] ^ *val;
	}
	else if (in->BmpInfo.bits == BIT8_3)
	{
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x)] ^ val[0];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 1] ^ val[1];
		in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] = in->BmpInfo.ImgData[(y * in->BmpInfo.width * 3) + (3 * x) + 2] ^ val[2];
	}
	return true;
}

bool ImageProcessing::SetMask(int mask)
{
	if (ImagInfo.BmpInfo.ImgData == NULL) return false;
	if (ImagInfo.BmpInfo.bits == BIT8_1)
	{
		for (int n = 0; n < ImagInfo.BmpInfo.imagesize; n++)
			ImagInfo.BmpInfo.ImgData[n] = ImagInfo.BmpInfo.ImgData[n] && mask;
	}
	else if (ImagInfo.BmpInfo.bits == BIT8_3)
	{
		for (int n = 0; n < ImagInfo.BmpInfo.height; n++)
		{
			for (int m = 0; m < ImagInfo.BmpInfo.width; m++)
			{
				ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m)] = ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m)] && mask;
				ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m) + 1] = ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m) + 1] && mask;
				ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m) + 2] = ImagInfo.BmpInfo.ImgData[(n * ImagInfo.BmpInfo.width * 3) + (3 * m) + 2] && mask;
			}
		}
	}
	return true;
}

//Color Mapping
bool ImageProcessing::Reverse(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	for (int n = 0; n < in.BmpInfo.height; n++)
	{
		for (int m = 0; m < in.BmpInfo.width; m++)
		{
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 255 - in.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 255 - in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255 - in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2];
		}
	}
	return true;
}

bool ImageProcessing::Grayscale(IMAGEINFOR in)
{
	int count = 0, gary = 0;
	if (in.BmpInfo.ImgData == NULL) return false;
	for (int n = 0; n < in.BmpInfo.imagesize; n++)
	{
		count++;
		switch (count)
		{
		case 1:
			gary += in.BmpInfo.ImgData[n] * 3736;
			break;
		case 2:
			gary += in.BmpInfo.ImgData[n] * 19235;
			break;
		case 3:
			gary += in.BmpInfo.ImgData[n] * 9797;
			gary = gary >> 15;
			in.BmpInfo.ImgData[n - 2] = gary; //B
			in.BmpInfo.ImgData[n - 1] = gary; //G
			in.BmpInfo.ImgData[n] = gary; //R
			count = 0;
			break;
		}
	}
	return true;
}

bool ImageProcessing::RGrayscale(IMAGEINFOR in)	//待更新
{
	int count = 0, gary = 0;
	if (in.BmpInfo.ImgData == NULL) return false;
	for (int n = 0; n < in.BmpInfo.imagesize; n++)
	{
		count++;
		switch (count)
		{
		case 1:
			gary += in.BmpInfo.ImgData[n] * 3736;
			break;
		case 2:
			gary += in.BmpInfo.ImgData[n] * 19235;
			break;
		case 3:
			gary += in.BmpInfo.ImgData[n] * 9797;
			gary = gary >> 15;
			in.BmpInfo.ImgData[n - 2] = gary; //B
			in.BmpInfo.ImgData[n - 1] = gary; //G
			in.BmpInfo.ImgData[n] = gary;	  //R
			count = 0;
			break;
		}
	}
	return true;
}

bool ImageProcessing::Binarization(IMAGEINFOR in)	//灰階二值化
{
	double gmean;
	Imgmath imgmath;
	if (in.BmpInfo.ImgData == NULL) return false;
	imgmath.Mean(in.BmpInfo.ImgData, &gmean);
	for (int n = 0; n < in.BmpInfo.imagesize; n++)
	{
		if (in.BmpInfo.ImgData[n] > gmean)
			in.BmpInfo.ImgData[n] = 255;
		else
			in.BmpInfo.ImgData[n] = 0;
	}
	return true;
}

bool ImageProcessing::Histogram(IMAGEINFOR in, double* histdata, int aisle, int shift)
{
	if (in.BmpInfo.ImgData == NULL || histdata == NULL) return false;
	int N = _msize(histdata) / sizeof(histdata[0]);
	memset(histdata, 0, N * sizeof(double));
	for (int n = 0; n < in.BmpInfo.width * in.BmpInfo.height; n++)
		histdata[in.BmpInfo.ImgData[n * aisle + shift]] = histdata[in.BmpInfo.ImgData[n * aisle + shift]] + 1;
	return true;
}

//Img應用
bool ImageProcessing::Turnto(IMAGEINFOR in)		//上下轉換
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int N = w * h * 3, count = 0;
	BYTE* buf = new BYTE[N];
	memset(buf, 0, N);
	for (int n = 0, m = N - 1; n < N, m >= 0; n++, m--)
	{
		count++;
		switch (count)
		{
		case 1:
			buf[n] = in.BmpInfo.ImgData[m - 2];
			break;
		case 2:
			buf[n] = in.BmpInfo.ImgData[m - 0];
			break;
		case 3:
			buf[n] = in.BmpInfo.ImgData[m + 2];
			count = 0;
			break;
		}
	}
	memcpy(in.BmpInfo.ImgData, buf, N);
	delete[]buf;
	return true;
}

bool ImageProcessing::Conversion(IMAGEINFOR in)	//左右顛倒
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int N = w * h * 3, count = 0;
	int sqrtnm = w, next = 0, Dre = sqrtnm * 3;
	BYTE* buf = new BYTE[N];
	memset(buf, 0, N);
	for (int n = 0; n < N; n++)
	{
		Dre--;
		count++;
		if (n % (sqrtnm * 3) == 0)
		{
			Dre = sqrtnm * 3;
			if ((n + sqrtnm * 3) < N)
				next += (sqrtnm * 3);
		}
		switch (count)
		{
		case 1:
			buf[n] = in.BmpInfo.ImgData[Dre - 1 - 2 + next];
			break;
		case 2:
			buf[n] = in.BmpInfo.ImgData[Dre - 1 - 0 + next];
			break;
		case 3:
			buf[n] = in.BmpInfo.ImgData[Dre - 1 + 2 + next];
			count = 0;
			break;
		}
	}
	memcpy(in.BmpInfo.ImgData, buf, N);
	delete[]buf;
	return true;
}

bool ImageProcessing::Stretch(IMAGEINFOR in, IMAGEINFOR out, int mul)		//拉寬
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int N   = in.BmpInfo.imagesize, next = 0;
	for (int n = 0; n < N; n++)
	{
		if (n != 0 && n % 3 == 0)
			next += 3 * (mul - 1);
		for (int nx = 0; nx < mul; nx++)
			out.BmpInfo.ImgData[n + next + 3 * nx] = in.BmpInfo.ImgData[n];
	}
	return true;
}

bool ImageProcessing::Pullup(IMAGEINFOR in, IMAGEINFOR out, int mul)		//拉高
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int N = in.BmpInfo.imagesize, next = 0;
	for (int n = 0; n < N; n++)
	{
		if (n != 0 && n % (w * 3) == 0)
			next++;
		for (int ny = 0; ny < mul; ny++)
			out.BmpInfo.ImgData[(n + (w * 3 * ny) + (next * (w * 3 * mul))) - (next * (w * 3))] = in.BmpInfo.ImgData[n];
	}
	return true;
}

bool ImageProcessing::Narrowing(IMAGEINFOR in, IMAGEINFOR out, int mul)	//縮寬
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int N = out.BmpInfo.imagesize, next = 0, val = 0;
	for (int n = 0; n < N; n++)
	{
		if (n != 0 && n % 3 == 0)
			next += 3 * (mul - 1);
		for (int nx = 0; nx < mul; nx++)
			val += in.BmpInfo.ImgData[n + next + 3 * nx];
		out.BmpInfo.ImgData[n] = val / mul;
		val = 0;
	}
	return true;
}

bool ImageProcessing::Shrink(IMAGEINFOR in, IMAGEINFOR out, int mul)		//縮高
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int N = out.BmpInfo.imagesize, next = 0, val = 0;
	for (int n = 0; n < N; n++)
	{
		if (n != 0 && n % (w * 3) == 0)
			next++;
		for (int ny = 0; ny < mul; ny++)
			val += in.BmpInfo.ImgData[(n + (w * 3 * ny) + (next * (w * 3 * mul))) - (next * (w * 3))];
		out.BmpInfo.ImgData[n] = val / mul;
		val = 0;
	}
	return true;
}

bool ImageProcessing::Scaling(IMAGEINFOR in, IMAGEINFOR out, int mul)		//縮放
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int N = in.BmpInfo.imagesize, nextw = 0, nexth = 0;
	int S = out.BmpInfo.imagesize, val = 0;
	if (N <= S)		//Enlarge
	{
		for (int n = 0; n < N; n++)
		{
			if (n != 0 && n % 3 == 0)		//next w
				nextw += 3 * (mul - 1);
			if (n != 0 && n % (w * 3) == 0)	//next h
				nexth++;
			for (int nx = 0; nx < mul; nx++)
				for (int ny = 0; ny < mul; ny++)
					out.BmpInfo.ImgData[(n + nextw + (3 * nx) + (w * 3 * mul * ny) + (nexth * (w * 3 * mul * mul))) - (nexth * (w * 3 * mul))] = in.BmpInfo.ImgData[n];
		}
	}
	else if (N > S)	//Shrink
	{
		for (int n = 0; n < S; n++)
		{
			if (n != 0 && n % 3 == 0)		//next w
				nextw += 3 * (mul - 1);
			if (n != 0 && n % (w * 3) == 0)	//next h
				nexth++;
			for (int nx = 0; nx < mul; nx++)
				for (int ny = 0; ny < mul; ny++)
					val += in.BmpInfo.ImgData[(n + nextw + (3 * nx) + (w * 3 * mul * ny) + (nexth * (w * 3 * mul * mul))) - (nexth * (w * 3 * mul))];
			out.BmpInfo.ImgData[n] = val / mul;
			val = 0;
		}
	}
	return true;
}

//bool ImageProcessing::Rotation(IMAGEINFOR in, unsigned char* buf, int n_w, int n_h, float theta)
bool ImageProcessing::Rotation(IMAGEINFOR in, IMAGEINFOR out, float theta)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int x, y, x_, y_, x_1, y_1, x_2, y_2;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	double sita = theta * M_PI / 180;
	double a = (w - 1) / 2;
	double b = (h - 1) / 2;
	//printf("sita = %.3f , a = %.3f  , b  = %.3f\n",sita,a,b);

	double x1 = -a * cos(sita) - b * sin(sita);
	double y1 = -a * sin(sita) + b * cos(sita);
	//printf("x1 = %.3f  , y1  = %.3f\n",x1,y1);

	double x2 = a * cos(sita) - b * sin(sita);
	double y2 = a * sin(sita) + b * cos(sita);
	//printf("x2 = %.3f  , y2  = %.3f\n",x2,y2);

	double x3 = a * cos(sita) + b * sin(sita);
	double y3 = a * sin(sita) - b * cos(sita);
	//printf("x3 = %.3f  , y3  = %.3f\n",x3,y3);

	double x4 = -a * cos(sita) + b * sin(sita);
	double y4 = -a * sin(sita) - b * cos(sita);
	//printf("x4 = %.3f  , y4  = %.3f\n",x4,y4);

	int w1 = round(MAX(abs(x1 - x3), abs(x2 - x4)));
	int h1 = round(MAX(abs(y1 - y3), abs(y2 - y4)));
	//printf("w1 = %d  , h1  = %d\n",w1,h1);

	double c = (w1 - 1) / 2;
	double d = (h1 - 1) / 2;
	//printf("c = %.3f  , d  = %.3f\n",c,d);

	double f1 = -c * cos(sita) + d * sin(sita) + a;
	double f2 = -c * sin(sita) - d * cos(sita) + b;
	//printf("f1 = %.3f  , f2  = %.3f\n",f1,f2);

	for (int n = 0; n < h1; n++)
	{
		for (int m = 0; m < w1; m++)
		{
			x = round(m * cos(sita) - n * sin(sita) + f1);
			y = round(m * sin(sita) + n * cos(sita) + f2);
			if (y >= 0 && y < h && x >= 0 && x < w)
			{
				x_ = ((n)+(h / 1)) * (w * 3 * 3);
				y_ = (m * 3) + ((w * 3 * 3) / 3);
				out.BmpInfo.ImgData[x_ + y_] = in.BmpInfo.ImgData[(y * w * 3) + (x * 3)];

				x_ = ((n)+(h / 1)) * (w * 3 * 3);
				y_ = (m * 3 + 1) + ((w * 3 * 3) / 3);
				out.BmpInfo.ImgData[x_ + y_] = in.BmpInfo.ImgData[(y * w * 3) + (x * 3 + 1)];

				x_ = ((n)+(h / 1)) * (w * 3 * 3);
				y_ = (m * 3 + 2) + ((w * 3 * 3) / 3);
				out.BmpInfo.ImgData[x_ + y_] = in.BmpInfo.ImgData[(y * w * 3) + (x * 3 + 2)];
			}
		}
	}
	return true;
}

bool ImageProcessing::Gridlines(IMAGEINFOR in, unsigned char* color, int cells)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	if (color == NULL)
	{
		color = new BYTE[3];
		color[0] = 0;
		color[1] = 0;
		color[2] = 0;
	}
	int nextw = round(w / cells), nexth = round(h / cells);
	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			if (m + 1 != w && n + 1 != h && m + 2 != w && n + 2 != h) 
			{
				if (n % nextw == 0 || m % nexth == 0) 
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = color[2];
					in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = color[1];
					in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = color[0];
				}
			}
		}
	}
	delete[]color;
	return true;
}

bool ImageProcessing::GaussInverseMatrix(double* A, double* B, int nColumns)	//反矩陣
{
	memcpy(B, A, sizeof(double) * nColumns * nColumns);
	int* is, * js, i, j, k, l, u, v;
	double d, p;
	is = new int[nColumns];
	js = new int[nColumns];
	for (k = 0; k <= nColumns - 1; k++)
	{
		d = 0.0;
		for (i = k; i <= nColumns - 1; i++)
		{
			for (j = k; j <= nColumns - 1; j++)
			{
				l = i * nColumns + j;
				p = fabs(B[l]);
				if (p > d)
				{
					d = p;
					is[k] = i;
					js[k] = j;
				}
			}
		}

		if (d + 1.0 == 1.0)
		{
			delete[]is;
			delete[]js;
			printf("err**not inv\n");
			return false;
		}

		if (is[k] != k)
		{
			for (j = 0; j <= nColumns - 1; j++)
			{
				u = k * nColumns + j;
				v = is[k] * nColumns + j;
				p = B[u];
				B[u] = B[v];
				B[v] = p;
			}
		}

		if (js[k] != k)
		{
			for (i = 0; i <= nColumns - 1; i++)
			{
				u = i * nColumns + k;
				v = i * nColumns + js[k];
				p = B[u];
				B[u] = B[v];
				B[v] = p;
			}
		}

		l = k * nColumns + k;
		B[l] = 1.0f / B[l];

		for (j = 0; j <= nColumns - 1; j++)
		{
			if (j != k)
			{
				u = k * nColumns + j;
				B[u] = B[u] * B[l];
			}
		}

		for (i = 0; i <= nColumns - 1; i++)
		{
			if (i != k)
			{
				for (j = 0; j <= nColumns - 1; j++)
				{
					if (j != k) {
						u = i * nColumns + j;
						B[u] -= B[i * nColumns + k] * B[k * nColumns + j];
					}
				}
			}
		}

		for (i = 0; i <= nColumns - 1; i++)
		{
			if (i != k)
			{
				u = i * nColumns + k;
				B[u] = -B[u] * B[l];
			}
		}
	}
	for (k = nColumns - 1; k >= 0; k--)
	{
		if (js[k] != k)
		{
			for (j = 0; j <= nColumns - 1; j++)
			{
				u = k * nColumns + j;
				v = js[k] * nColumns + j;
				p = B[u];
				B[u] = B[v];
				B[v] = p;
			}
		}

		if (is[k] != k)
		{
			for (i = 0; i <= nColumns - 1; i++)
			{
				u = i * nColumns + k;
				v = i * nColumns + is[k];
				p = B[u];
				B[u] = B[v];
				B[v] = p;
			}
		}
	}
	delete[]is;
	delete[]js;
	return true;
}

void ImageProcessing::matrixMul(double* a, double* b, int m, int n, int k, double* c)
{
	int i, j, l, u;
	for (i = 0; i < m; i++)
	{
		for (j = 0; j < k; j++)
		{
			u = i * k + j;
			c[u] = 0.0;
			for (l = 0; l < n; l++)
				c[u] += a[i * n + l] * b[l * k + j];
		}
	}
}

bool ImageProcessing::f_MLSR(IMAGEINFOR in, int* srcP, int* dstP, int pNum)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE)* w * h * 3);

	int i, j, n, m;
	double* srcPx = new double[pNum];
	double* srcPy = new double[pNum];
	double* dstPx = new double[pNum];
	double* dstPy = new double[pNum];

	double* GammaIJ = new double[pNum * pNum];		//Γij  = e^-||p-Xi|| / β^2
	double* GammaIJT = new double[pNum * pNum];		//Γijt = Γij + λ * (W^-1)
	double* GammaP = new double[pNum];				//Γp   = e^-||p-Xi|| / β^2
	double* S = new double[pNum];
	double* W = new double[pNum * pNum];
	double* WInv = new double[pNum * pNum];
	double* tempIJ = new double[pNum * pNum];

	double belta = 2.0;	//β
	double alpha = 2.0;	//α
	double lambda = 1.0;	//λ
	double fbelta = 1.0 / (belta * belta); // 1/β^2

	//設置控制點
	for (i = 0; i < pNum; i++)
	{
		srcPx[i] = (double)srcP[2 * i] / w;
		srcPy[i] = (double)srcP[2 * i + 1] / h;
		dstPx[i] = (double)dstP[2 * i] / w;
		dstPy[i] = (double)dstP[2 * i + 1] / h;
	}

	//Γij = e^-||p-Xi|| / β^2 , exp() - e^..
	int wp = pNum;
	for (j = 0; j < pNum; j++)
		for (i = 0; i < pNum; i++)
			GammaIJ[j * wp + i] = exp(-((double)(srcPx[i] - srcPx[j]) * (srcPx[i] - srcPx[j]) + (srcPy[i] - srcPy[j]) * (srcPy[i] - srcPy[j])) * fbelta);

	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			//權重計算Wi
			double px = (double)i / w, py = (double)j / h;
			for (n = 0; n < pNum; n++)
			{
				for (m = 0; m < pNum; m++)
				{
					if (n == m)	//Wi(p) = ||p-Xi|| ^ -2α , pow() - 平方
						W[n * wp + m] = pow(((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])), -alpha);
					else
						W[n * wp + m] = 0;
				}
			}

			//計算W的逆矩陣
			GaussInverseMatrix(W, WInv, pNum);

			//計算Gamma + lambda * WInv
			for (n = 0; n < pNum; n++)
				for (m = 0; m < pNum; m++)	//Γijt = Γij + λ * (W^-1)
					GammaIJT[n * wp + m] = GammaIJ[n * wp + m] + lambda * WInv[n * wp + m];

			//計算 ( Gamma + lambda * WInv ) 的逆矩陣
			GaussInverseMatrix(GammaIJT, tempIJ, pNum);

			//計算 GammaP ( Γp = e^-||p-Xi||/β^2 ) 
			for (m = 0; m < pNum; m++)
				GammaP[m] = exp((-((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])) * fbelta));

			//計算 S = Γp * tempIJ
			matrixMul(GammaP, tempIJ, 1, pNum, pNum, S);

			//計算 f(p) = T + SY
			double sumx = 0, sumy = 0;
			for (m = 0; m < pNum; m++)
			{
				sumx += S[m] * srcPx[m];
				sumy += S[m] * srcPy[m];
			}
			px = px - sumx;
			py = py - sumy;

			sumx = 0, sumy = 0;
			for (m = 0; m < pNum; m++)
			{
				sumx += S[m] * dstPx[m];
				sumy += S[m] * dstPy[m];
			}
			px = px + sumx;
			py = py + sumy;

			//轉換完新座標
			int xx = (int)CLIP3(px * w, 0, w - 1);
			int yy = (int)CLIP3(py * h, 0, h - 1);

			in.BmpInfo.ImgData[(j * w * 3) + (3 * i)] = buf[(yy * w * 3) + (3 * xx)];
			in.BmpInfo.ImgData[(j * w * 3) + ((3 * i) + 1)] = buf[(yy * w * 3) + ((3 * xx) + 1)];
			in.BmpInfo.ImgData[(j * w * 3) + ((3 * i) + 2)] = buf[(yy * w * 3) + ((3 * xx) + 2)];
		}
	}
	delete[]srcPx;
	delete[]srcPy;
	delete[]dstPx;
	delete[]dstPy;
	delete[]GammaIJ;
	delete[]GammaP;
	delete[]S;
	delete[]W;
	delete[]WInv;
	delete[]tempIJ;
	delete[]GammaIJT;
	return true;
}

bool ImageProcessing::MLSR(IMAGEINFOR in, int* srcP, int* dstP, int pNum)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE)* w * h * 3);

	int i, j, n, m;
	double* srcPx = new double[pNum];
	double* srcPy = new double[pNum];
	double* dstPx = new double[pNum];
	double* dstPy = new double[pNum];

	double* GammaIJ = new double[pNum * pNum];		//Γij  = e^-||p-Xi|| / β^2
	double* GammaIJT = new double[pNum * pNum];		//Γijt = Γij + λ * (W^-1)
	double* GammaP = new double[pNum];				//Γp   = e^-||p-Xi|| / β^2
	double* S = new double[pNum];
	double* W = new double[pNum * pNum];
	double* WInv = new double[pNum * pNum];
	double* tempIJ = new double[pNum * pNum];

	double belta = 2.0;	//β
	double alpha = 2.0;	//α
	double lambda = 1.0;	//λ
	double fbelta = 1.0 / (belta * belta); // 1/β^2

	//設置控制點
	for (i = 0; i < pNum; i++)
	{
		srcPx[i] = (double)srcP[2 * i] / w;
		srcPy[i] = (double)srcP[2 * i + 1] / h;
		dstPx[i] = (double)dstP[2 * i] / w;
		dstPy[i] = (double)dstP[2 * i + 1] / h;
		//printf("srcPx = %f , srcPy = %f , dstPx = %f , dstPy = %f\n",srcPx[i],srcPy[i],dstPx[i],dstPy[i]);
	}

	//Γij = e^-||p-Xi|| / β^2 , exp() - e^..
	int wp = pNum;
	for (j = 0; j < pNum; j++)
	{
		for (i = 0; i < pNum; i++)
		{
			GammaIJ[j * wp + i] = exp(-((double)(srcPx[i] - srcPx[j]) * (srcPx[i] - srcPx[j]) + (srcPy[i] - srcPy[j]) * (srcPy[i] - srcPy[j])) * fbelta);
			//printf("GammaIJ = %f\n",GammaIJ[j*wp + i]);
		}
	}

	for (j = 0; j < h; j++)
	{
		for (i = 0; i < w; i++)
		{
			//權重計算Wi
			double px = (double)i / w, py = (double)j / h;
			//if(j == 1)
				//printf("px = %.3f , py = %.3f\n",px,py);

			for (n = 0; n < pNum; n++)
			{
				for (m = 0; m < pNum; m++)
				{
					if (n == m)	//Wi(p) = ||p-Xi|| ^ -2α , pow() - 平方
						W[n * wp + m] = pow(((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])), -alpha);
					else
						W[n * wp + m] = 0;

					//if(j == 1)
						//printf("W = %.3f\n",W[n*wp + m]);
				}
			}

			//計算W的逆矩陣
			GaussInverseMatrix(W, WInv, pNum);

			//計算Gamma + lambda * WInv
			for (n = 0; n < pNum; n++)
			{
				for (m = 0; m < pNum; m++)
				{
					//Γijt = Γij + λ * (W^-1)
					GammaIJT[n * wp + m] = GammaIJ[n * wp + m] + lambda * WInv[n * wp + m];
					//printf("GammaIJT = %.3f\n",GammaIJT[n*wp + m]);
				}
			}

			/*if(j == 1)
				for(n = 0; n < pNum; n++)
					for(m = 0; m < pNum; m++)
						printf("GammaIJT[%d] = %.3f\n",n*wp + m,GammaIJT[n*wp + m]);*/

						//計算 ( Gamma + lambda * WInv ) 的逆矩陣
			GaussInverseMatrix(GammaIJT, tempIJ, pNum);

			/*if(j == 1)
				for(n = 0; n < pNum; n++)
					for(m = 0; m < pNum; m++)
						printf("tempIJ[%d] = %.3f\n",n*wp + m,tempIJ[n*wp + m]);*/

						//計算 GammaP ( Γp = e^-||p-Xi||/β^2 ) 
			for (m = 0; m < pNum; m++)
				GammaP[m] = exp((-((double)(px - srcPx[m]) * (px - srcPx[m]) + (py - srcPy[m]) * (py - srcPy[m])) * fbelta));

			//計算 S = Γp * tempIJ
			matrixMul(GammaP, tempIJ, 1, pNum, pNum, S);

			if (j == 1)
			{
				//for(m = 0; m < pNum; m++)
					//printf("S[%d] = %.3f\n",m,S[m]);
			}

			//計算 f(p) = T + SY
			double sumx = 0, sumy = 0;
			for (m = 0; m < pNum; m++)
			{
				sumx += S[m] * srcPx[m];
				sumy += S[m] * srcPy[m];
			}
			px = px - sumx;
			py = py - sumy;

			//if(j == 1)
				//printf("[1] px = %.3f , py = %.3f\n",px,py);

			sumx = 0, sumy = 0;
			for (m = 0; m < pNum; m++)
			{
				sumx += S[m] * dstPx[m];
				sumy += S[m] * dstPy[m];
			}
			px = px + sumx;
			py = py + sumy;

			//if(j == 1)
				//printf("[2] px = %.3f , py = %.3f\n",px,py);

			//轉換完新座標
			int xx = (int)CLIP3(px * w, 0, w - 1);
			int yy = (int)CLIP3(py * h, 0, h - 1);

			//if(j == 1)
				//printf("xx = %d , yy = %d\n",xx,yy);

			in.BmpInfo.ImgData[(j * w * 3) + (3 * i)] = buf[(yy * w * 3) + (3 * xx)];
			in.BmpInfo.ImgData[(j * w * 3) + ((3 * i) + 1)] = buf[(yy * w * 3) + ((3 * xx) + 1)];
			in.BmpInfo.ImgData[(j * w * 3) + ((3 * i) + 2)] = buf[(yy * w * 3) + ((3 * xx) + 2)];
		}
	}
	delete[]srcPx;
	delete[]srcPy;
	delete[]dstPx;
	delete[]dstPy;
	delete[]GammaIJ;
	delete[]GammaP;
	delete[]S;
	delete[]W;
	delete[]WInv;
	delete[]tempIJ;
	delete[]GammaIJT;
	delete[]buf;
	return true;
}

bool ImageProcessing::DistortingMirror(IMAGEINFOR in, int x, int y, int k)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	double theta;
	int dx, dy, R, R_, cdx, cdy;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			dx = m - x;
			dy = n - y;
			theta = atan2((double)dy, (double)dx);	//反正切函數
			R   = (int)sqrt((double)(pow(dx, 2) + pow(dy, 2)));
			R_  = (int)sqrt((double)R * k);
			cdx = x + (int)(R_ * cos(theta));
			cdy = y + (int)(R_ * sin(theta));
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	    = buf[(cdy * w * 3) + (3 * cdx)];
			in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = buf[(cdy * w * 3) + ((3 * cdx) + 1)];
			in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = buf[(cdy * w * 3) + ((3 * cdx) + 2)];
		}
	}
	delete[]buf;
	return true;
}

//Filter
bool ImageProcessing::S_Convolution(IMAGEINFOR in, float* kernel)	//一般速
{
	float val;
	int n, m, x, y, kx;

	int N = _msize(kernel) / sizeof(kernel[0]);
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
		for (m = 0; m < in.BmpInfo.width * 3; m++)
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + m + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + m];

	for (x = 0; x <in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width * 3; y++)
		{
			val = 0;
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				val += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + y + ((kx - nextW) * 3)] * kernel[kx];
			}

			if (val > 255) val = 255;
			else if (val < 0) val = 0;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + y] = val;
			shiftW = 0; next = 1; nextW = 0;
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Convolution(IMAGEINFOR in, float* kernel)	//加速
{
	float Rval, Gval, Bval;
	int n, m, x, y, kx;

	int N = _msize(kernel) / sizeof(kernel[0]);
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
	{
		for (m = 0; m < in.BmpInfo.width; m++)
		{
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + (3 * m) + (3 * shift)]       = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + (3 * m)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 1) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 1)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 2) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 2)];
		}
	}

	for (x = 0; x < in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width; y++)
		{
			Rval = 0; Gval = 0; Bval = 0;
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				Rval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + (3 * y) + ((kx - nextW) * 3)] * kernel[kx];
				Gval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 1) + ((kx - nextW) * 3)] * kernel[kx];
				Bval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 2) + ((kx - nextW) * 3)] * kernel[kx];
			}

			if (Rval >= 255 || Gval >= 255 || Bval >= 255)
			{
				Rval = 255; Gval = 255; Bval = 255;
			}
			else if (Rval <= 0 || Gval <= 0 || Bval <= 0)
			{
				Rval = 0; Gval = 0; Bval = 0;
			}

			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + (3 * y)] = Rval;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = Gval;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = Bval;
			shiftW = 0; next = 1; nextW = 0;
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Sharpness(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	kernel[0] = -1; kernel[1] = -1; kernel[2] = -1;
	kernel[3] = -1; kernel[4] = 9; kernel[5] = -1;
	kernel[6] = -1; kernel[7] = -1; kernel[8] = -1;

	/*kernel[0] = 1;kernel[1] = 1;kernel[2] = 1;
	kernel[3] = 1;kernel[4] = -7;kernel[5] = 1;
	kernel[6] = 1;kernel[7] = 1;kernel[8] = 1;*/

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::Blur_BoxFilter(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	/*kernel[0] = 0;kernel[1] = 0.2;kernel[2] = 0;
	kernel[3] = 0.2;kernel[4] = 0;kernel[5] = 0.2;
	kernel[6] = 0;kernel[7] = 0.2;kernel[8] = 0;*/

	/*kernel[0]  = 0;kernel[1]  = 0;kernel[2]  = 0.077;kernel[3]  = 0;kernel[4]  = 0;
	kernel[5]  = 0;kernel[6]  = 0.077;kernel[7]  =  0.077;kernel[8]  = 0.077;kernel[9]  = 0;
	kernel[10] = 0.077;kernel[11] = 0.077;kernel[12] =  0.077;kernel[13] = 0.077;kernel[14] = 0.077;
	kernel[15] = 0;kernel[16] = 0.077;kernel[17] =  0.077;kernel[18] = 0.077;kernel[19] = 0;
	kernel[20] = 0;kernel[21] = 0;kernel[22] =  0.077;kernel[23] = 0;kernel[24] = 0;*/

	kernel[0] = 0.11; kernel[1] = 0.11; kernel[2] = 0.11;
	kernel[3] = 0.11; kernel[4] = 0.11; kernel[5] = 0.11;
	kernel[6] = 0.11; kernel[7] = 0.11; kernel[8] = 0.11;

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::Blur_GaussianFilter(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	kernel[0] = 0.045; kernel[1] = 0.122; kernel[2] = 0.045;
	kernel[3] = 0.122; kernel[4] = 0.332; kernel[5] = 0.122;
	kernel[6] = 0.045; kernel[7] = 0.122; kernel[8] = 0.045;

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::Blur_Motion(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[25];
	/*kernel[0] = 1; kernel[1] = 0; kernel[2] = 0;
	kernel[3] = 0; kernel[4] = 1; kernel[5] = 0;
	kernel[6] = 0; kernel[7] = 0; kernel[8] = 1;*/

	kernel[0] = 0.2; kernel[1] = 0; kernel[2] = 0; kernel[3] = 0; kernel[4] = 0;
	kernel[5] = 0; kernel[6] = 0.2; kernel[7] = 0; kernel[8] = 0; kernel[9] = 0;
	kernel[10] = 0; kernel[11] = 0; kernel[12] = 0.2; kernel[13] = 0; kernel[14] = 0;
	kernel[15] = 0; kernel[16] = 0; kernel[17] = 0; kernel[18] = 0.2; kernel[19] = 0;
	kernel[20] = 0; kernel[21] = 0; kernel[22] = 0; kernel[23] = 0; kernel[24] = 0.2;

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::MedianFilter(IMAGEINFOR in) //中值濾波器 ( 降噪 )
{
	float Rval, Gval, Bval;
	int n, m, x, y, kx;
	if (in.BmpInfo.ImgData == NULL) return false;

	float* rkernel = new float[9];
	float* gkernel = new float[9];
	float* bkernel = new float[9];
	int N = _msize(rkernel) / sizeof(rkernel[0]);
	memset(rkernel, 0, N * sizeof(float));
	memset(gkernel, 0, N * sizeof(float));
	memset(bkernel, 0, N * sizeof(float));

	Sort sort;
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
	{
		for (m = 0; m < in.BmpInfo.width; m++)
		{
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + (3 * m) + (3 * shift)]       = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + (3 * m)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 1) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 1)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 2) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 2)];
		}
	}

	for (x = 0; x < in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width; y++)
		{
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				bkernel[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + (3 * y) + ((kx - nextW) * 3)];
				gkernel[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 1) + ((kx - nextW) * 3)];
				rkernel[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 2) + ((kx - nextW) * 3)];
			}
			sort.InsertionSort(bkernel);
			sort.InsertionSort(gkernel);
			sort.InsertionSort(rkernel);
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + (3 * y)]       = bkernel[N / 2];
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = gkernel[N / 2];
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = rkernel[N / 2];
			shiftW = 0; next = 1; nextW = 0;
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::FuzzyMedianFilter(IMAGEINFOR in) //中值濾波器 ( 降噪 ) 待修正
{
	float val = 0.0;
	int n, m, x, y, kx;
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	int N = _msize(kernel) / sizeof(kernel[0]);
	memset(kernel, 0, N * sizeof(float));

	Sort sort;
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
		for (m = 0; m < in.BmpInfo.width * 3; m++)
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + m + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + m];

	for (x = 0; x < in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width * 3; y++)
		{
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				kernel[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + y + ((kx - nextW) * 3)];
			}
			sort.InsertionSort(kernel);
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + y] = kernel[N / 2];
			shiftW = 0; next = 1; nextW = 0;
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::PrewittFilter(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	Grayscale(in);
	float* kr = new float[9];
	kr[0] = 1; kr[1] = 0; kr[2] = -1;
	kr[0] = 1; kr[1] = 0; kr[2] = -1;
	kr[0] = 1; kr[1] = 0; kr[2] = -1;
	Convolution(in, kr);

	kr[0] = -1; kr[1] = -1; kr[2] = -1;
	kr[0] = 0;	kr[1] = 0;	kr[2] = 0;
	kr[0] = 1;	kr[1] = 1;	kr[2] = 1;
	Convolution(in, kr);
	return true;
}

bool ImageProcessing::LoGFilter(IMAGEINFOR in, double sa, int krn)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	double sa2, sa4, nor, nx;
	int n, m, ct = -1, w = ImagInfo.BmpInfo.width, h = ImagInfo.BmpInfo.height;
	float* kr = new float[krn * krn];
	Grayscale(in);
	nx = (krn - 1) / 2;
	for (n = nx; n > -nx - 1; n--)
	{
		for (m = -nx; m < nx + 1; m++)
		{
			ct++;
			sa2 = pow(sa, 1.0 * 2);
			sa4 = pow(sa, 1.0 * 4);
			nor = pow(m, 1.0 * 2) + pow(n, 1.0 * 2);
			kr[ct] = ((nor / sa4 - 2 / sa2) * exp(-nor / (2 * sa2)));
		}
	}
	Convolution(in, kr);
	return true;
}

bool ImageProcessing::DoGFilter(IMAGEINFOR in, double sa, double sb, int krn)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	double sa2, nor, nx;
	int n, m, ct = -1, w = in.BmpInfo.width, h = in.BmpInfo.height;
	float* kr = new float[krn * krn];

	Grayscale(in);

	IMAGEINFOR ab,bb;
	ab.BmpInfo.width  = w;
	ab.BmpInfo.height = h;
	ab.BmpInfo.ImgData = new BYTE[w * h * 3];
	memcpy(ab.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	bb.BmpInfo.width  = w;
	bb.BmpInfo.height = h;
	bb.BmpInfo.ImgData = new BYTE[w * h * 3];
	memcpy(bb.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	nx = (krn - 1) / 2;
	for (n = nx; n > -nx - 1; n--)	//sigma1
	{
		for (m = -nx; m < nx + 1; m++)
		{
			ct++;
			sa2 = pow(sa, 1.0 * 2);
			nor = pow(m, 1.0 * 2) + pow(n, 1.0 * 2);
			kr[ct] = (1 / (2 * M_PI * sa2)) * exp(-nor / (2 * sa2));
		}
	}
	Convolution(ab, kr);

	ct = -1;
	for (n = nx; n > -nx - 1; n--)	//sigma2
	{
		for (m = -nx; m < nx + 1; m++)
		{
			ct++;
			sa2 = pow(sb, 1.0 * 2);
			nor = pow(m, 1.0 * 2) + pow(n, 1.0 * 2);
			kr[ct] = (1 / (2 * M_PI * sa2)) * exp(-nor / (2 * sa2));
		}
	}
	Convolution(bb, kr);

	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			ab.BmpInfo.ImgData[(n * w * 3) + (3 * m)]     = bb.BmpInfo.ImgData[(n * w * 3) + (3 * m)]     - ab.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
			ab.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = bb.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] - ab.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1];
			ab.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = bb.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] - ab.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2];
		}
	}
	memcpy(in.BmpInfo.ImgData, ab.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);
	delete[]ab.BmpInfo.ImgData;
	delete[]bb.BmpInfo.ImgData;
	return true;
}

bool ImageProcessing::EdgeDetection_Sobelfilter(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	kernel[0] = -1; kernel[1] = -1; kernel[2] = -1;
	kernel[3] = -1; kernel[4] = 8; kernel[5] = -1;
	kernel[6] = -1; kernel[7] = -1; kernel[8] = -1;

	//left
	/*kernel[0] = -1;kernel[1] = -2;kernel[2] = -1;
	kernel[3] =  0;kernel[4] =  0;kernel[5] =  0;
	kernel[6] =  1;kernel[7] =  2;kernel[8] =  1;*/

	//Right
	/*kernel[0] = 1; kernel[1] = 0; kernel[2] = -1;
	kernel[3] = 2; kernel[4] = 0; kernel[5] = -2;
	kernel[6] = 1; kernel[7] = 0; kernel[8] = -1;*/

	//Top
	/*kernel[0] = 1; kernel[1] = 2; kernel[2] = 1;
	kernel[3] = 0; kernel[4] = 0; kernel[5] = 0;
	kernel[6] = -1; kernel[7] = -2; kernel[8] = -1;*/

	//Bottom
	/*kernel[0] = -1; kernel[1] = 0; kernel[2] = 1;
	kernel[3] = -2; kernel[4] = 0; kernel[5] = 2;
	kernel[6] = -1; kernel[7] = 0; kernel[8] = 1;*/

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::EdgeDetection_LaplacianEdgeDetector(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	/*kernel[0] =  0;kernel[1] =  1;kernel[2] =  0;
	kernel[3] =  1;Kernel[4] = -4;kernel[5] =  1;
	kernel[6] =  0;kernel[7] =  1;kernel[8] =  0;*/

	kernel[0] = 1; kernel[1] = 1; kernel[2] = 1;
	kernel[3] = 1; kernel[4] = -8; kernel[5] = 1;
	kernel[6] = 1; kernel[7] = 1; kernel[8] = 1;

	Convolution(in, kernel);
	return true;
}

bool ImageProcessing::Embossing(IMAGEINFOR in)
{
	int n, m, x, y, kx, Rval, Gval, Bval;
	if (in.BmpInfo.ImgData == NULL) return false;

	float* kernel = new float[9];
	kernel[0] = -1; kernel[1] = -1; kernel[2] = 0;
	kernel[3] = -1; kernel[4] = 0; kernel[5] = 1;
	kernel[6] = 0; kernel[7] = 1; kernel[8] = 1;

	int N = _msize(kernel) / sizeof(kernel[0]);
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
	{
		for (m = 0; m < in.BmpInfo.width; m++)
		{
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + (3 * m) + (3 * shift)]       = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + (3 * m)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 1) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 1)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 2) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 2)];
		}
	}

	for (x = 0; x < in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width; y++)
		{
			Rval = 0; Gval = 0; Bval = 0;
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				Rval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + (3 * y) + ((kx - nextW) * 3)] * kernel[kx];
				Gval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 1) + ((kx - nextW) * 3)] * kernel[kx];
				Bval += buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 2) + ((kx - nextW) * 3)] * kernel[kx];
			}

			if (Rval >= 255 || Gval >= 255 || Bval >= 255)
			{
				Rval = 255; Gval = 255; Bval = 255;
			}
			else if (Rval <= 0 || Gval <= 0 || Bval <= 0)
			{
				Rval = 0; Gval = 0; Bval = 0;
			}

			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + (3 * y)] = Rval + 128;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = Gval + 128;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = Bval + 128;
			shiftW = 0; next = 1; nextW = 0;
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::KMeansCluster(IMAGEINFOR in, int k)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int n, m, kn, temp, sum, sv;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	BYTE* kmeanbuf = new BYTE[k];
	memset(kmeanbuf, 0, sizeof(BYTE) * k);

	int* countbuf = new int[k];
	memset(countbuf, 0, sizeof(int) * k);

	int* addbuf = new int[k];
	memset(addbuf, 0, sizeof(int) * k);

	int* rbuf = new int[k];
	memset(rbuf, 0, sizeof(int) * k);

	int* gbuf = new int[k];
	memset(gbuf, 0, sizeof(int) * k);

	int* bbuf = new int[k];
	memset(bbuf, 0, sizeof(int) * k);

	BYTE* jdugebuf = new BYTE[k];
	memset(jdugebuf, 0, sizeof(BYTE) * k);

	BYTE* markbuf = new BYTE[w * h * 3];
	memset(markbuf, 0, sizeof(BYTE) * w * h * 3);

	IMAGEINFOR b;
	b.BmpInfo.ImgData = new BYTE[w * h * 3];
	b.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(b.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	bool s = true;
	Grayscale(b);

	while (s)
	{
		sum = 0;
		for (kn = 0; kn < k; kn++)
			kmeanbuf[kn] = (kn * 255) / (k - 1);

		for (n = 0; n < h; n++)
		{
			for (m = 0; m < w; m++)
			{
				temp = abs(b.BmpInfo.ImgData[(n * w * 3) + (3 * m)] - kmeanbuf[0]);	//b color
				sv = 0;
				for (kn = 0; kn < k; kn++)
				{
					int t = abs(b.BmpInfo.ImgData[(n * w * 3) + (3 * m)] - kmeanbuf[kn]);
					if (temp > t)
					{
						temp = t;
						sv = kn;
					}
				}
				countbuf[sv]++;
				addbuf[sv] += b.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
				markbuf[(n * w * 3) + (3 * m)] = sv;
				markbuf[(n * w * 3) + ((3 * m) + 1)] = sv;
				markbuf[(n * w * 3) + ((3 * m) + 2)] = sv;
			}
		}

		for (kn = 0; kn < k; kn++)
		{
			if (addbuf[kn] == 0 || countbuf[kn] == 0)
				kmeanbuf[kn] = 0;
			else
				kmeanbuf[kn] = addbuf[kn] / countbuf[kn];
			sum += kmeanbuf[kn] - jdugebuf[kn];
			jdugebuf[kn] = kmeanbuf[kn];
		}

		if (sum == 0)
			s = false;
	}

	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			bbuf[markbuf[(n * w * 3) + (3 * m)]]	   += in.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
			gbuf[markbuf[(n * w * 3) + ((3 * m) + 1)]] += in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)];
			rbuf[markbuf[(n * w * 3) + ((3 * m) + 2)]] += in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)];
		}
	}

	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]       = bbuf[markbuf[(n * w * 3) + (3 * m)]] / countbuf[markbuf[(n * w * 3) + (3 * m)]];
			in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = gbuf[markbuf[(n * w * 3) + ((3 * m) + 1)]] / countbuf[markbuf[(n * w * 3) + ((3 * m) + 1)]];
			in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = rbuf[markbuf[(n * w * 3) + ((3 * m) + 2)]] / countbuf[markbuf[(n * w * 3) + ((3 * m) + 2)]];
		}
	}

	delete[]b.BmpInfo.ImgData;
	delete[]markbuf;
	delete[]jdugebuf;
	delete[]bbuf;
	delete[]gbuf;
	delete[]rbuf;
	delete[]addbuf;
	delete[]countbuf;
	delete[]kmeanbuf;
	return true;
}

bool ImageProcessing::Haar(IMAGEINFOR in, int depth)	//哈爾小波轉換 ( Harr )
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int count = 1;
	int radd, gadd, badd, rless, gless, bless;
	int n, m, kx, wf, hf;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	Grayscale(in);
	//Sharpness(in);

	BYTE* buf = new BYTE[w * h * 3];
	memset(buf, 0, sizeof(BYTE) * w * h * 3);

	BYTE* ibuf = new BYTE[w * h * 3];
	memcpy(ibuf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	while (depth >= count)
	{
		wf = w / count;
		hf = h / count;
		printf("wf = %d , hf = %d\n", wf, hf);

		//行
		for (n = 0; n < hf; n++)
		{
			for (m = 0; m < wf / 2; m++)
			{
				buf[(n * w * 3) + (3 * m)] = (ibuf[(n * w * 3) + (3 * 2 * m)] + ibuf[(n * w * 3) + ((3 * 2 * m) + 3)]) / 2;	//sqrt(2)
				buf[(n * w * 3) + ((3 * m) + 1)] = (ibuf[(n * w * 3) + ((3 * 2 * m) + 1)] + ibuf[(n * w * 3) + ((3 * 2 * m) + 1) + 3]) / 2;
				buf[(n * w * 3) + ((3 * m) + 2)] = (ibuf[(n * w * 3) + ((3 * 2 * m) + 2)] + ibuf[(n * w * 3) + ((3 * 2 * m) + 2) + 3]) / 2;

				bless = (ibuf[(n * w * 3) + (3 * 2 * m)] - ibuf[(n * w * 3) + (3 * 2 * m) + 3]) / 2;
				gless = (ibuf[(n * w * 3) + (3 * 2 * m) + 1] - ibuf[(n * w * 3) + (3 * 2 * m) + 1 + 3]) / 2;
				rless = (ibuf[(n * w * 3) + (3 * 2 * m) + 2] - ibuf[(n * w * 3) + (3 * 2 * m) + 2 + 3]) / 2;

				if (bless < 0 || gless < 0 || rless < 0)
				{
					bless = 0;
					gless = 0;
					rless = 0;
				}

				buf[(n * w * 3) + (3 * m) + (((wf / 2)) * 3)] = bless;
				buf[(n * w * 3) + ((3 * m) + 1) + (((wf / 2)) * 3)] = gless;
				buf[(n * w * 3) + ((3 * m) + 2) + (((wf / 2)) * 3)] = rless;
			}
		}

		radd = 0; gadd = 0; badd = 0;
		rless = 0; gless = 0; bless = 0;

		//列
		for (n = 0; n < hf / 2; n++)
		{
			for (m = 0; m < wf; m++)
			{
				ibuf[(n * w * 3) + (3 * m)] = (buf[(n * w * 3 * 2) + (3 * m)] + buf[(n * w * 3 * 2) + (3 * m) + (w * 3)]) / 2;
				ibuf[(n * w * 3) + ((3 * m) + 1)] = (buf[(n * w * 3 * 2) + ((3 * m) + 1)] + buf[(n * w * 3 * 2) + ((3 * m) + 1) + (w * 3)]) / 2;
				ibuf[(n * w * 3) + ((3 * m) + 2)] = (buf[(n * w * 3 * 2) + ((3 * m) + 2)] + buf[(n * w * 3 * 2) + ((3 * m) + 2) + (w * 3)]) / 2;

				bless = (buf[(n * w * 3 * 2) + (3 * m)] - buf[(n * w * 3 * 2) + (3 * m) + (w * 3)]) / 2;
				gless = (buf[(n * w * 3 * 2) + ((3 * m) + 1)] - buf[(n * w * 3 * 2) + ((3 * m) + 1) + (w * 3)]) / 2;
				rless = (buf[(n * w * 3 * 2) + ((3 * m) + 2)] - buf[(n * w * 3 * 2) + ((3 * m) + 2) + (w * 3)]) / 2;

				if (bless < 0 || gless < 0 || rless < 0)
				{
					bless = 0;
					gless = 0;
					rless = 0;
				}

				ibuf[((n + ((hf / 2))) * w * 3) + (3 * m)] = bless;
				ibuf[((n + ((hf / 2))) * w * 3) + ((3 * m) + 1)] = gless;
				ibuf[((n + ((hf / 2))) * w * 3) + ((3 * m) + 2)] = rless;
			}
		}
		count++;
	}

	memcpy(in.BmpInfo.ImgData, ibuf, sizeof(BYTE) * w * h * 3);

	delete[]buf;
	delete[]ibuf;
	return true;
}

bool ImageProcessing::CAfilter(IMAGEINFOR in, int w, int h)	//圓形濾波器 ( 待修正 )
{
	int n, m, x, y, kx, Rval, Gval, Bval;
	if (in.BmpInfo.ImgData == NULL) return false;

	int* srcTemp = new int[9];
	BYTE* buf = new BYTE[w * h * 3];

	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	for (n = 1; n < h - 1; n++)
	{
		for (m = 1; m < w - 1; m++)
		{
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = buf[(n * w) + m] * 1;
			in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m - 1))] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m - 1)) + 1] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m - 1)) + 2] = buf[((n - 1) * w) + (m - 1)] * 1.41;
			in.BmpInfo.ImgData[(n * w * 3) + (3 * (m - 1))] = in.BmpInfo.ImgData[(n * w * 3) + (3 * (m - 1)) + 1] = in.BmpInfo.ImgData[(n * w * 3) + (3 * (m - 1)) + 2] = buf[(n * w) + (m - 1)] * 1;

			in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m - 1))] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m - 1)) + 1] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m - 1)) + 2] = buf[((n + 1) * w) + (m - 1)] * 0.654;
			in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * m)] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * m) + 1] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * m) + 2] = buf[((n + 1) * w) + m] * 1;
			in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m + 1))] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m + 1)) + 1] = in.BmpInfo.ImgData[((n + 1) * w * 3) + (3 * (m + 1)) + 2] = buf[((n + 1) * w) + (m + 1)] * 3.64;

			in.BmpInfo.ImgData[(n * w * 3) + (3 * (m + 1))] = in.BmpInfo.ImgData[(n * w * 3) + (3 * (m + 1)) + 1] = in.BmpInfo.ImgData[(n * w * 3) + (3 * (m + 1)) + 2] = buf[(n * w) + (m + 1)] * 1;
			in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m + 1))] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m + 1)) + 1] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * (m + 1)) + 2] = buf[((n - 1) * w) + (m + 1)] * 0.86625;
			in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * m)] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * m) + 1] = in.BmpInfo.ImgData[((n - 1) * w * 3) + (3 * m) + 2] = buf[((n - 1) * w) + m] * 1;

			if (n == 1)
			{
				//printf("0 = %d | 1 = %d | 2 = %d\n", (n * w * 3) + (3 * m), ((n - 1) * w * 3) + (3 * (m - 1)), (n * w * 3) + (3 * (m - 1)));
				//printf("3 = %d | 4 = %d | 5 = %d\n", ((n + 1) * w * 3) + (3 * (m - 1)), ((n + 1) * w * 3) + (3 * m), ((n + 1) * w * 3) + (3 * (m + 1)));
				//printf("6 = %d | 7 = %d | 8 = %d\n", (n * w * 3) + (3 * (m + 1)), ((n - 1) * w * 3) + (3 * (m + 1)), ((n - 1) * w * 3) + (3 * m));
			}
		}
	}

	delete[]buf;
	return true;
}

int ImageProcessing::CountN(int* src)
{
	int N = _msize(src) / sizeof(src[0]);
	int count = 0;
	for (int i = 0; i < N; i++)
	{
		if (src[i] == 0)
		{
			count++;
		}
	}
	return count;
}

int ImageProcessing::CountT(int* src)
{
	int count = 0;
	int N = _msize(src) / sizeof(src[0]);
	for (int i = 1; i < N; i++)
	{
		if (src[i] == 1 && src[i - 1] == 0)
		{
			count++;
		}
	}
	if (src[N - 1] == 0 && src[0] == 1)
	{
		count++;
	}
	return count;
}

bool ImageProcessing::Thinning_Zhang(IMAGEINFOR in)		//骨架化
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	int* srcTemp = new int[9];
	int countNumber;
	do
	{
		countNumber = 0;
		for (int n = 1; n < h - 1; n++)
		{
			for (int m = 1; m < w - 1; m++)
			{
				srcTemp[0] = in.BmpInfo.ImgData[(n * w) + m];
				srcTemp[1] = in.BmpInfo.ImgData[((n - 1) * w) + m];
				srcTemp[2] = in.BmpInfo.ImgData[((n - 1) * w) + (m - 1)];
				srcTemp[3] = in.BmpInfo.ImgData[(n * w) + (m - 1)];
				srcTemp[4] = in.BmpInfo.ImgData[((n + 1) * w) + (m - 1)];
				srcTemp[5] = in.BmpInfo.ImgData[((n + 1) * w) + m];
				srcTemp[6] = in.BmpInfo.ImgData[((n + 1) * w) + (m + 1)];
				srcTemp[7] = in.BmpInfo.ImgData[(n * w) + (m + 1)];
				srcTemp[8] = in.BmpInfo.ImgData[((n - 1) * w) + (m + 1)];

				//printf("srcTemp[0] = %d , srcTemp[1] = %d , srcTemp[2] = %d\n",inptr[(n*w*3) + (3*m)],inptr[((n-1) * w * 3) + (3 * (m-1))],inptr[(n * w * 3) + (3 * (m-1))]);
				//printf("srcTemp[3] = %d , srcTemp[4] = %d , srcTemp[5] = %d\n",inptr[((n+1) * w * 3) + (3 * (m-1))],inptr[((n+1) * w * 3) + (3 * m)],inptr[((n+1) * w * 3) + (3 * (m+1))]);
				//printf("srcTemp[6] = %d , srcTemp[7] = %d , srcTemp[8] = %d\n",inptr[(n * w * 3) + (3 * (m+1))],inptr[((n-1) * w * 3) + (3 * (m+1))],inptr[((n-1) * w * 3) + (3 * m)]);

				if (in.BmpInfo.ImgData[(n * w) + m] != 1)
				{
					if (CountN(srcTemp) >= 2 && CountN(srcTemp) <= 6)
					{
						if (CountT(srcTemp) == 1)
						{
							if (in.BmpInfo.ImgData[(n * w) + (m - 1)] * in.BmpInfo.ImgData[((n + 1) * w) + m] * in.BmpInfo.ImgData[(n * w) + (m + 1)] == 0)
							{
								if (in.BmpInfo.ImgData[((n - 1) * w) + m] * in.BmpInfo.ImgData[((n + 1) * w) + m] * in.BmpInfo.ImgData[(n * w) + (m + 1)] == 0)
								{
									in.BmpInfo.ImgData[(n * w) + m] = (BYTE)1;
									countNumber++;
									//printf("countNumber = %d\n",countNumber);
								}
							}
							else
							{
								if (in.BmpInfo.ImgData[(n * w) + (m - 1)] * in.BmpInfo.ImgData[((n + 1) * w) + m] * in.BmpInfo.ImgData[((n - 1) * w) + m] == 0)
								{
									if (in.BmpInfo.ImgData[(n * w) + (m - 1)] * in.BmpInfo.ImgData[(n * w) + (m + 1)] * in.BmpInfo.ImgData[((n - 1) * w) + m] == 0)
									{
										in.BmpInfo.ImgData[(n * w) + m] = (BYTE)1;
										countNumber++;
										//printf("countNumber = %d\n",countNumber);
									}
								}
							}
						}
					}
				}
			}
		}
	} while (countNumber != 0);
	return true;
}

bool ImageProcessing::Segmentation(IMAGEINFOR in)
{
	int n, m;
	double gmean;
	Imgmath imgmath;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* gbuf = new BYTE[w * h * 3];
	memcpy(gbuf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	BYTE* iibuf = new BYTE[w * h * 3];
	memcpy(iibuf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	IMAGEINFOR b,ib;
	b.BmpInfo.width  = w;
	b.BmpInfo.height = h;
	b.BmpInfo.ImgData = new BYTE[w * h * 3];

	ib.BmpInfo.width  = w;
	ib.BmpInfo.height = h;
	ib.BmpInfo.ImgData   = new BYTE[w * h * 3];
	ib.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(ib.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	Grayscale(ib);				//灰階
	Blur_GaussianFilter(ib);	//高斯模糊

	for (n = 0; n < h; n++)		//Pixels differences ( 待修正 )
	{
		for (m = 0; m < w - 1; m++)
		{
			//gbuf[(n * w * 3) + (3 * m)]       = (ibuf[(n * w * 3) + (3 * 2 * m)]       - ibuf[(n * w * 3) + ((3 * 2 * m) + 3)]);
			//gbuf[(n * w * 3) + ((3 * m) + 1)] = (ibuf[(n * w * 3) + ((3 * 2 * m) + 1)] - ibuf[(n * w * 3) + ((3 * 2 * m) + 1) + 3]);
			//gbuf[(n * w * 3) + ((3 * m) + 2)] = (ibuf[(n * w * 3) + ((3 * 2 * m) + 2)] - ibuf[(n * w * 3) + ((3 * 2 * m) + 2) + 3]);

			gbuf[(n * w * 3) + (3 * m)]		  = (ib.BmpInfo.ImgData[(n * w * 3) + (3 * m)]		 + ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 3)]) / sqrt(2);
			gbuf[(n * w * 3) + ((3 * m) + 1)] = (ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] + ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1) + 3]) / sqrt(2);
			gbuf[(n * w * 3) + ((3 * m) + 2)] = (ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] + ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2) + 3]) / sqrt(2);
		}
	}

	for (n = 0; n < h - 1; n++)
	{
		for (m = 0; m < w; m++)
		{
			//ibuf[(n * w * 3) + (3 * m)]       = (gbuf[(n * w * 3 * 2) + (3 * m)]       - gbuf[(n * w * 3 * 2) + (3 * m) + (w * 3)]);
			//ibuf[(n * w * 3) + ((3 * m) + 1)] = (gbuf[(n * w * 3 * 2) + ((3 * m) + 1)] - gbuf[(n * w * 3 * 2) + ((3 * m) + 1) + (w * 3)]);
			//ibuf[(n * w * 3) + ((3 * m) + 2)] = (gbuf[(n * w * 3 * 2) + ((3 * m) + 2)] - gbuf[(n * w * 3 * 2) + ((3 * m) + 2) + (w * 3)]);

			ib.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	    = (gbuf[(n * w * 3) + (3 * m)] + gbuf[(n * w * 3) + (3 * m) + (w * 3)]) / sqrt(2);
			ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = (gbuf[(n * w * 3) + ((3 * m) + 1)] + gbuf[(n * w * 3) + ((3 * m) + 1) + (w * 3)]) / sqrt(2);
			ib.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = (gbuf[(n * w * 3) + ((3 * m) + 2)] + gbuf[(n * w * 3) + ((3 * m) + 2) + (w * 3)]) / sqrt(2);
		}
	}

	memcpy(in.BmpInfo.ImgData, ib.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	//CAfilter(inptr, w, h);	//圓形平均濾波 ( 詢問Paper內容 )
	MedianFilter(in);
	MedianFilter(in);
	MedianFilter(in);
	MedianFilter(in);
	MedianFilter(in);

	for (n = 0; n < h; n++)
		for (m = 0; m < w; m++)
			b.BmpInfo.ImgData[(n * w) + m] = (in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] + in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] + in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)]) / 3;

	imgmath.Mean(b.BmpInfo.ImgData, &gmean);			//二值化
	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			if (b.BmpInfo.ImgData[(n * w) + m] > gmean)
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = 0;
			else
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = 255;
		}
	}

	for (n = 0; n < h; n++)
		for (m = 0; m < w; m++)
			b.BmpInfo.ImgData[(n * w) + m] = (in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] * 0.114 + in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] * 0.587 + in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] * 0.299 < 128 ? 0 : 1);

	Thinning_Zhang(b);	//圖像骨架化

	for (n = 0; n < h; n++)
		for (m = 0; m < w; m++)
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 1)] = in.BmpInfo.ImgData[(n * w * 3) + ((3 * m) + 2)] = 255 - (b.BmpInfo.ImgData[(n * w) + m] * 255);

	for (n = 0; n < h; n++)		//把邊緣寫回原圖
	{
		for (m = 0; m < w; m++)
		{
			if (in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] == 255)
				iibuf[(n * w * 3) + (3 * m)] = iibuf[(n * w * 3) + (3 * m) + 1] = iibuf[(n * w * 3) + (3 * m) + 2] = 255;
		}
	}

	memcpy(in.BmpInfo.ImgData, iibuf, sizeof(BYTE) * w * h * 3);

	delete[]b.BmpInfo.ImgData;
	delete[]gbuf;
	//delete[]ibuf;
	//delete[]iibuf;
	return true;
}

bool ImageProcessing::Dilation(IMAGEINFOR in, int w, int h)	//膨脹
{
	if (in.BmpInfo.ImgData == NULL) return false;
	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			if (m == 0 || m == w - 1 || n == 0 || n == h - 1)
			{
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 255;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 255;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255;
			}
			else
			{
				if (buf[m * 3 - 3 + n * w * 3] == 255 || buf[m * 3 + n * w * 3] == 255 || buf[m * 3 + 3 + n * w * 3] == 255 || buf[m * 3 + (n - 1) * w * 3] == 255 || buf[m * 3 + (n + 1) * w * 3] == 255)
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 255;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 255;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255;
				}
				else
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 0;
				}
			}
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Corrosion(IMAGEINFOR in, int w, int h)	//侵蝕
{
	if (in.BmpInfo.ImgData == NULL) return false;
	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			if (m == 0 || m == w - 1 || n == 0 || n == h - 1)
			{
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 255;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 255;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255;
			}
			else
			{
				if (buf[m * 3 - 3 + n * w * 3] == 255 && buf[m * 3 + n * w * 3] == 255 && buf[m * 3 + 3 + n * w * 3] == 255 && buf[m * 3 + (n - 1) * w * 3] == 255 && buf[m * 3 + (n + 1) * w * 3] == 255)
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 255;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 255;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255;
				}
				else
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 0;
				}
			}
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Matting(IMAGEINFOR in, int w, int h)	//待更新
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int n;

	Grayscale(in);		//灰階
	Binarization(in);	//灰階二值化

	Imagematting sm;
	sm.loadImage(in.BmpInfo.ImgData, w, h);			//原圖
	sm.loadTrimap(in.BmpInfo.ImgData, w, h);		//Trimap圖
	sm.solveAlpha();								//生成alpha圖

	/*for (n = 0; n < 1; n++)
		Dilation(in);	//膨脹

	for (n = 0; n < 1; n++)
		Corrosion(in);	//侵蝕

	for (n = 0; n < 1; n++)
		Dilation(in);	//膨脹
	//Corrosion(in);	//侵蝕*/

	return true;
}

bool ImageProcessing::Blending(IMAGEINFOR in)	//合成
{


	return true;
}

bool ImageProcessing::Inpainting(IMAGEINFOR in)	//修補
{


	return true;
}

bool ImageProcessing::Completion(IMAGEINFOR in)	//補全
{


	return true;
}

bool ImageProcessing::Retargeting(IMAGEINFOR in)	//重新定位。縮小圖片長寬，裁剪多餘像素，但是保留景物原本形狀。
{


	return true;
}

//特徵擷取 , 判斷
bool ImageProcessing::ImgRange(IMAGEINFOR in, IMAGEINFOR out, float* knl, int xst, int yst)
{
	int shift = -1, next = -1, TCount = 0;
	if (in.BmpInfo.ImgData == NULL) return false;
	int N = _msize(knl) / sizeof(knl[0]);
	int Square = sqrt(N), xs = xst * 3, ys = yst * in.BmpInfo.width * 3;

	for (int run = 0; run < Square * Square * 3; run++)
	{
		shift++;
		if (run == 0 || run % 3 == 0) 
		{
			next++;
			knl[next] = in.BmpInfo.ImgData[shift + xs + ys];
		}

		if (shift == Square * 3) 
		{
			shift = 0;
			ys += (in.BmpInfo.width * 3 - 1);
		}
		out.BmpInfo.ImgData[run] = in.BmpInfo.ImgData[shift + xs + ys];
	}
	return true;
}

bool ImageProcessing::TemplateMatching(IMAGEINFOR in, float* tel, int method)
{
	int n, m, x, y, kx, i;
	if (in.BmpInfo.ImgData == NULL) return false;

	int N = _msize(tel) / sizeof(tel[0]);
	int ker = sqrt(N) - 1;
	int shift = ker / 2;
	int shiftW = 0, next = 1, nextW = 0;

	float Rval, Gval, Bval;
	float rSa, rSb, rCosine, rMax, gSa, gSb, gCosine, gMax, bSa, bSb, bCosine, bMax;
	float* rbuf = new float[N];
	float* gbuf = new float[N];
	float* bbuf = new float[N];
	memset(rbuf, 0, N * sizeof(float));
	memset(gbuf, 0, N * sizeof(float));
	memset(bbuf, 0, N * sizeof(float));

	printf("N = %d , ker = %d , shift = %d\n", N, ker, shift);

	BYTE* buf = new BYTE[((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3];
	memset(buf, 0, ((in.BmpInfo.width + ker) * (in.BmpInfo.height + ker)) * 3);

	float* dbuf = new float[in.BmpInfo.width * in.BmpInfo.height * 3];
	memset(dbuf, 0, in.BmpInfo.width * in.BmpInfo.height * 3);

	for (n = 0; n < in.BmpInfo.height; n++)
	{
		for (m = 0; m < in.BmpInfo.width; m++)
		{
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + (3 * m) + (3 * shift)]		  = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + (3 * m)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 1) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 1)];
			buf[((n + shift) * ((in.BmpInfo.width + ker) * 3)) + ((3 * m) + 2) + (3 * shift)] = in.BmpInfo.ImgData[(n * (in.BmpInfo.width * 3)) + ((3 * m) + 2)];
		}
	}

	for (x = 0; x < in.BmpInfo.height; x++)
	{
		for (y = 0; y < in.BmpInfo.width; y++)
		{
			Rval = 0; Gval = 0; Bval = 0;
			for (kx = 0; kx < N; kx++)
			{
				if (kx != 0 && (kx / next) % (ker + 1) == 0)
				{
					next++;
					nextW = kx;
					shiftW += (in.BmpInfo.width + ker) * 3;
				}
				bbuf[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + (3 * y) + ((kx - nextW) * 3)];
				gbuf[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 1) + ((kx - nextW) * 3)];
				rbuf[kx] = buf[(x * ((in.BmpInfo.width + ker) * 3)) + shiftW + ((3 * y) + 2) + ((kx - nextW) * 3)];

				/*if (x == 0 && y == 0)
				{
					//printf("kx[%d] , Next = %d , NextW = %d , ShiftW = %d\n",kx,Next,NextW,ShiftW);
					//printf("[%d] Buf = %.3f , buf = ( %d )\n",(x * ((BmpInfo.width + Ker) * 3)) + ShiftW + y + ((kx - NextW) * 3),Buf[kx],BufBmpInfo.ImgData[(x * ((BmpInfo.width + Ker) * 3)) + ShiftW + y + ((kx - NextW) * 3)]);
				}*/
			}

			switch (method)
			{
			case MN_COSINE_SIMILARITY:		//餘弦相似度
				for (i = 0; i < N; i++)
				{
					bCosine += bbuf[i] * tel[i];
					bSa += bbuf[i] * bbuf[i];
					bSb += tel[i] * tel[i];

					gCosine += gbuf[i] * tel[i];
					gSa += gbuf[i] * gbuf[i];
					gSb += tel[i] * tel[i];

					rCosine += rbuf[i] * tel[i];
					rSa += rbuf[i] * rbuf[i];
					rSb += tel[i] * tel[i];
				}
				bSa = sqrt(bSa);
				bSb = sqrt(bSb);
				bCosine /= (bSa * bSb);

				gSa = sqrt(gSa);
				gSb = sqrt(gSb);
				gCosine /= (gSa * gSb);

				rSa = sqrt(rSa);
				rSb = sqrt(rSb);
				rCosine /= (rSa * rSb);

				if (bCosine > bMax) bMax = bCosine;
				if (gCosine > gMax) gMax = gCosine;
				if (rCosine > rMax) rMax = rCosine;

				Bval = -1 * (bCosine * 255);
				Gval = -1 * (gCosine * 255);
				Rval = -1 * (rCosine * 255);

				dbuf[(x * (in.BmpInfo.width * 3)) + (3 * y)] = bCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = gCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = rCosine;
				break;
			case MN_SQDIFF:					//平方差匹配
				for (i = 0; i < N; i++)
				{
					Bval += (bbuf[i] - tel[i]) * (bbuf[i] - tel[i]);
					Gval += (gbuf[i] - tel[i]) * (gbuf[i] - tel[i]);
					Rval += (rbuf[i] - tel[i]) * (rbuf[i] - tel[i]);
				}
				break;
			case MN_SQDIFF_NORMED:			//標準平方差匹配
				for (i = 0; i < N; i++)
				{
					bCosine += (bbuf[i] - tel[i]) * (bbuf[i] - tel[i]);
					bSa += bbuf[i] * bbuf[i];
					bSb += tel[i] * tel[i];

					gCosine += (gbuf[i] - tel[i]) * (gbuf[i] - tel[i]);
					gSa += gbuf[i] * gbuf[i];
					gSb += tel[i] * tel[i];

					rCosine += (rbuf[i] - tel[i]) * (rbuf[i] - tel[i]);
					rSa += rbuf[i] * rbuf[i];
					rSb += tel[i] * tel[i];
				}
				bSa = sqrt(bSa);
				bSb = sqrt(bSb);
				bCosine /= (bSa * bSb);

				gSa = sqrt(gSa);
				gSb = sqrt(gSb);
				gCosine /= (gSa * gSb);

				rSa = sqrt(rSa);
				rSb = sqrt(rSb);
				rCosine /= (rSa * rSb);

				if (bCosine > bMax) bMax = bCosine;
				if (gCosine > gMax) gMax = gCosine;
				if (rCosine > rMax) rMax = rCosine;

				Bval = -1 * (bCosine * 255);
				Gval = -1 * (gCosine * 255);
				Rval = -1 * (rCosine * 255);

				dbuf[(x * (in.BmpInfo.width * 3)) + (3 * y)] = bCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = gCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = rCosine;
				break;
			case MN_CCORR:					//相關匹配
				for (i = 0; i < N; i++)
				{
					Bval += (bbuf[i] * tel[i]);
					Gval += (gbuf[i] * tel[i]);
					Rval += (rbuf[i] * tel[i]);
				}
				break;
			case MN_CCORR_NORMED:			//標準相關匹配 ( 餘弦相似度 )
				for (i = 0; i < N; i++)
				{
					bCosine += bbuf[i] * tel[i];
					bSa += bbuf[i] * bbuf[i];
					bSb += tel[i] * tel[i];

					gCosine += gbuf[i] * tel[i];
					gSa += gbuf[i] * gbuf[i];
					gSb += tel[i] * tel[i];

					rCosine += rbuf[i] * tel[i];
					rSa += rbuf[i] * rbuf[i];
					rSb += tel[i] * tel[i];
				}
				bSa = sqrt(bSa);
				bSb = sqrt(bSb);
				bCosine /= (bSa * bSb);

				gSa = sqrt(gSa);
				gSb = sqrt(gSb);
				gCosine /= (gSa * gSb);

				rSa = sqrt(rSa);
				rSb = sqrt(rSb);
				rCosine /= (rSa * rSb);

				if (bCosine > bMax) bMax = bCosine;
				if (gCosine > gMax) gMax = gCosine;
				if (rCosine > rMax) rMax = rCosine;

				Bval = -1 * (bCosine * 255);
				Gval = -1 * (gCosine * 255);
				Rval = -1 * (rCosine * 255);

				dbuf[(x * (in.BmpInfo.width * 3)) + (3 * y)] = bCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = gCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = rCosine;
				break;
			case MN_CCOEFF:					//相關匹配 ( 均值 )
				for (i = 0; i < N; i++)
				{
					bSa += bbuf[i];
					bSb += tel[i];

					gSa += gbuf[i];
					gSb += tel[i];

					rSa += rbuf[i];
					rSb += tel[i];
				}
				bSa = bSa / N;
				bSb = bSb / N;
				Bval = bSa * bSb;

				gSa = gSa / N;
				gSb = gSb / N;
				Gval = gSa * gSb;

				rSa = rSa / N;
				rSb = rSb / N;
				Rval = rSa * rSb;
				break;
			case MN_CCOEFF_NORMED:			//標準相關匹配 ( 餘弦相似度 , 均值 ) - 待更新
				for (i = 0; i < N; i++)
				{
					bCosine += bbuf[i] * tel[i];
					bSa += bbuf[i] * bbuf[i];
					bSb += tel[i] * tel[i];

					gCosine += gbuf[i] * tel[i];
					gSa += gbuf[i] * gbuf[i];
					gSb += tel[i] * tel[i];

					rCosine += rbuf[i] * tel[i];
					rSa += rbuf[i] * rbuf[i];
					rSb += tel[i] * tel[i];
				}
				bSa = sqrt(bSa);
				bSb = sqrt(bSb);
				bCosine /= (bSa * bSb);

				gSa = sqrt(gSa);
				gSb = sqrt(gSb);
				gCosine /= (gSa * gSb);

				rSa = sqrt(rSa);
				rSb = sqrt(rSb);
				rCosine /= (rSa * rSb);

				if (bCosine > bMax) bMax = bCosine;
				if (gCosine > gMax) gMax = gCosine;
				if (rCosine > rMax) rMax = rCosine;

				Bval = -1 * (bCosine * 255);
				Gval = -1 * (gCosine * 255);
				Rval = -1 * (rCosine * 255);

				dbuf[(x * (in.BmpInfo.width * 3)) + (3 * y)] = bCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = gCosine;
				dbuf[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = rCosine;
				break;
			}
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + (3 * y)] = Bval;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 1)] = Gval;
			in.BmpInfo.ImgData[(x * (in.BmpInfo.width * 3)) + ((3 * y) + 2)] = Rval;

			shiftW = 0; next = 1; nextW = 0;
			bSa = 0.0; bSb = 0.0; bCosine = 0.0;
			gSa = 0.0; gSb = 0.0; gCosine = 0.0;
			rSa = 0.0; rSb = 0.0; rCosine = 0.0;
		}
	}
	//printf("rMax = %.3f\n", Max);
	for (int Mn = 0; Mn < in.BmpInfo.imagesize; Mn++)
		if (dbuf[Mn] == rMax)
			in.BmpInfo.ImgData[Mn] = 255;
	delete[]rbuf;
	delete[]gbuf;
	delete[]bbuf;
	delete[]buf;
	delete[]dbuf;
	return true;;
}

//Image Painting
bool ImageProcessing::Oilpainting(IMAGEINFOR in)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int k, dx, dy;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			k = rand() % 2;
			dx = (m + k) >= w ? w - 1 : (m + k);
			dy = (n + k) >= h ? h - 1 : (n + k);
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = buf[(dy * w * 3) + (3 * dx)];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = buf[(dy * w * 3) + (3 * dx) + 1];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = buf[(dy * w * 3) + (3 * dx) + 2];
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Atomization(IMAGEINFOR in, int v)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int k, dx, dy;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			k = rand() % v;
			dx = (m + k) >= w ? w - 1 : (m + k);
			dy = (n + k) >= h ? h - 1 : (n + k);
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = buf[(dy * w * 3) + (3 * dx)];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = buf[(dy * w * 3) + (3 * dx) + 1];
			in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = buf[(dy * w * 3) + (3 * dx) + 2];
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Mosaic(IMAGEINFOR in, int v)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int vb, vg, vr, dx, dy;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			dx = n;
			dy = m;
			if (n % v == 0)
			{
				if (m % v == 0)
				{
					vb = buf[(dx * w * 3) + (3 * dy)];
					vg = buf[(dx * w * 3) + (3 * dy) + 1];
					vr = buf[(dx * w * 3) + (3 * dy) + 2];
				}
				else
				{
					in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy)]		= (BYTE)vb;
					in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy) + 1] = (BYTE)vg;
					in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy) + 2] = (BYTE)vr;
				}
			}
			else
			{
				in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy)]		= in.BmpInfo.ImgData[((dx - 1) * w * 3) + (3 * dy)];
				in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy) + 1] = in.BmpInfo.ImgData[((dx - 1) * w * 3) + (3 * dy) + 1];
				in.BmpInfo.ImgData[(dx * w * 3) + (3 * dy) + 2] = in.BmpInfo.ImgData[((dx - 1) * w * 3) + (3 * dy) + 2];
			}
		}
	}
	delete[]buf;
	return true;
}

// 座標 x y 半徑 r 亮度 thd
bool ImageProcessing::Sunlight(IMAGEINFOR in, int x, int y, int r, float thd)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int vb, vg, vr;
	float cr, pv;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	BYTE* buf = new BYTE[w * h * 3];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	if (x >= w || y >= h || x < 0 || y < 0)
	{
		x = w / 2;
		y = h / 2;
	}

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			vb = buf[(n * w * 3) + (3 * m)];
			vg = buf[(n * w * 3) + (3 * m) + 1];
			vr = buf[(n * w * 3) + (3 * m) + 2];
			cr = (float)sqrt(pow(m - x, 2) + pow(n - y, 2));
			if (cr < r)
			{
				pv = thd * (1.0 - cr / r);
				vb = vb + pv;
				vg = vg + pv;
				vr = vr + pv;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = (BYTE)(vb > 0 ? (vb < 255 ? vb : 255) : 0);
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = (BYTE)(vg > 0 ? (vg < 255 ? vg : 255) : 0);
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = (BYTE)(vr > 0 ? (vr < 255 ? vr : 255) : 0);
				vb = 0; vg = 0; vr = 0;
			}
		}
	}
	delete[]buf;
	return true;
}

bool ImageProcessing::Noise(IMAGEINFOR in, BYTE fn, double mv, double vv)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	double v1 = 0, v2 = 0, tmpv = 0;
	BYTE* buf = new BYTE[in.BmpInfo.imagesize];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE)* in.BmpInfo.imagesize);

	for (int n = 0; n < in.BmpInfo.imagesize; n++)
	{
		switch (fn)
		{
		case 0:	//Gauss noise 0 20
			do
			{
				v1 = (double)rand() / (RAND_MAX + 1.0);
			} while (v1 <= 0.00000000001);
			v2 = (double)rand() / (RAND_MAX + 1.0);
			tmpv = sqrt(-2 * log(v1)) * cos(2 * M_PI * v2) * vv + mv;
			break;
		case 1:	//Rayleigh noise 0 200
			do
			{
				v1 = (double)rand() / (RAND_MAX + 1.0);
			} while (v1 >= 0.9999999999);
			tmpv = mv + sqrt(-1 * vv * log(1 - v1));
			break;
		case 2:	//Noise index  0.1
			do
			{
				v1 = (double)rand() / (RAND_MAX + 1.0);
			} while (v1 >= 0.9999999999);
			tmpv = -1 * log(1 - v1) / mv;
			break;
		case 3:	//Salt and pepper noise 0.02 0.02
			v1 = (double)rand() / (RAND_MAX + 1.0);
			if (v1 <= mv)
				tmpv = -500;
			else if (v1 >= (1 - vv))
				tmpv = 500;
			else
				tmpv = 0;
			break;
		default:
			break;
		}
		tmpv = tmpv + buf[n];

		if (tmpv > 255)
			buf[n] = 255;
		else if (tmpv < 0)
			buf[n] = 0;
		else
			buf[n] = (int)tmpv;
	}
	memcpy(in.BmpInfo.ImgData, buf, sizeof(BYTE)* in.BmpInfo.imagesize);
	delete[]buf;
	return true;
}

bool ImageProcessing::FloodFill(IMAGEINFOR in, TPoint p, RGBColor fcolor, int thd)
{
	if (in.BmpInfo.ImgData == NULL) return false;

	int gray, xn = 0, yn = 0, xnn = 0, ynn = 0, w = in.BmpInfo.width, h = in.BmpInfo.height;
	int* mask = new int[w * h];

	BYTE* buf = new BYTE[in.BmpInfo.imagesize];
	memcpy(buf, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	RGBColor bcolor;
	bcolor.bval = buf[(p.y * w * 3) + (3 * p.x)];
	bcolor.gval = buf[(p.y * w * 3) + (3 * p.x) + 1];
	bcolor.rval = buf[(p.y * w * 3) + (3 * p.x) + 2];
	gray = (int)((bcolor.bval + bcolor.gval + bcolor.rval) / 3);

	if (p.x < 0 || p.x >= w || p.y < 0 || p.y >= h) return false;
	xn = p.x + 1;
	yn = p.y;
	int ct = w * h;
	while (ct > 0)
	{
		xnn = xn = xn - 1;
		if (ct % w == 0) ynn = yn - 1;
		if (xn < 0) xnn = xn = 0;
		if (yn < 0) ynn = yn = 1;
		mask[(ynn * w) + (xnn)] = 1;
		buf[(ynn * w * 3) + (3 * xnn)] = fcolor.bval;
		buf[(ynn * w * 3) + (3 * xnn) + 1] = fcolor.gval;
		buf[(ynn * w * 3) + (3 * xnn) + 2] = fcolor.rval;
		if (xn > 0 && (abs(gray - (int)((buf[(ynn * w * 3) + (3 * (xnn - 1))] +
			buf[(ynn * w * 3) + (3 * (xnn - 1)) + 1] +
			buf[(ynn * w * 3) + (3 * (xnn - 1)) + 2]) / 3)) < thd) && (mask[(ynn * w) + (xnn - 1)] != 1))
		{
			buf[(ynn * w * 3) + (3 * (xnn - 1))] = fcolor.bval;
			buf[(ynn * w * 3) + (3 * (xnn - 1)) + 1] = fcolor.gval;
			buf[(ynn * w * 3) + (3 * (xnn - 1)) + 2] = fcolor.rval;
			xn = (xnn + 1) - 1;
			yn = ynn;
			mask[(ynn * w) + (xnn - 1)] = 1;
		}
		if (xn < w - 1 && (abs(gray - (int)((buf[(ynn * w * 3) + (3 * (xnn + 1))] +
			buf[(ynn * w * 3) + (3 * (xnn + 1)) + 1] +
			buf[(ynn * w * 3) + (3 * (xnn + 1)) + 2]) / 3)) < thd) && (mask[(ynn * w) + (xnn + 1)] != 1))
		{
			buf[(ynn * w * 3) + (3 * (xnn + 1))] = fcolor.bval;
			buf[(ynn * w * 3) + (3 * (xnn + 1)) + 1] = fcolor.gval;
			buf[(ynn * w * 3) + (3 * (xnn + 1)) + 2] = fcolor.rval;
			xn = (xnn + 1) + 1;
			yn = ynn;
			mask[(ynn * w) + (xnn + 1)] = 1;
		}
		if (yn > 0 && (abs(gray - (int)((buf[((ynn - 1) * w * 3) + (3 * xnn)] +
			buf[((ynn - 1) * w * 3) + (3 * xnn) + 1] +
			buf[((ynn - 1) * w * 3) + (3 * xnn) + 2]) / 3)) < thd) && (mask[((ynn - 1) * w) + xnn] != 1))
		{
			buf[((ynn - 1) * w * 3) + (3 * xnn)] = fcolor.bval;
			buf[((ynn - 1) * w * 3) + (3 * xnn) + 1] = fcolor.gval;
			buf[((ynn - 1) * w * 3) + (3 * xnn) + 2] = fcolor.rval;
			xn = (xnn + 1);
			yn = ynn - 1;
			mask[((ynn - 1) * w) + xnn] = 1;
		}
		if (yn < h - 1 && (abs(gray - (int)((buf[((ynn + 1) * w * 3) + (3 * xnn)] +
			buf[((ynn + 1) * w * 3) + (3 * xnn) + 1] +
			buf[((ynn + 1) * w * 3) + (3 * xnn) + 2]) / 3)) < thd) && (mask[((ynn + 1) * w) + xnn] != 1))
		{
			buf[((ynn + 1) * w * 3) + (3 * xnn)] = fcolor.bval;
			buf[((ynn + 1) * w * 3) + (3 * xnn) + 1] = fcolor.gval;
			buf[((ynn + 1) * w * 3) + (3 * xnn) + 2] = fcolor.rval;
			xn = (xnn + 1);
			yn = ynn + 1;
			mask[((ynn + 1) * w) + xnn] = 1;
		}
		ct--;
	}
	memcpy(in.BmpInfo.ImgData, buf, sizeof(BYTE) * w * h * 3);
	delete[]buf;
	return true;
}

bool ImageProcessing::FFTImage(IMAGEINFOR in, int mode)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	IMAGEINFOR b;
	b.BmpInfo.imagesize = in.BmpInfo.imagesize;
	b.BmpInfo.ImgData   = new BYTE[in.BmpInfo.imagesize];
	memcpy(b.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);
	Grayscale(b);
	DFT dft = DFT(b.BmpInfo.ImgData, w, h);
	if (mode == 0) memcpy(in.BmpInfo.ImgData, dft.FFTImg(), sizeof(BYTE) * w * h * 3);		//幅度函数
	if (mode == 1) memcpy(in.BmpInfo.ImgData, dft.FFTPhaseImg(), sizeof(BYTE) * w * h * 3);	//相位函数
	delete[]b.BmpInfo.ImgData;
	return true;
}

//Image Understanding
//face
bool ImageProcessing::SkinDetect(IMAGEINFOR in)	//膚色檢測
{
	if (in.BmpInfo.ImgData == NULL) return false;

	double r, g, b;
	int R, G, B, S, w = in.BmpInfo.width, h = in.BmpInfo.height;

	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			B = in.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
			G = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1];
			R = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2];
			r = (double)R / (double)(R + G + B + 1);
			g = (double)G / (double)(R + G + B + 1);
			b = (double)B / (double)(R + G + B + 1);
			S = R + G + B;
			if (S != 0)
			{
				if (((double)(95.0 / (double)S) < r) && (r < (1 - (double)(90.0 / (double)S))) && (g > (double)(50.0 / (double)S)) && ((r - g) > (double)(30.0 / (double)S)) && ((r - g) < 0.2))
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)]	  = (BYTE)B;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = (BYTE)G;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = (BYTE)R;
				}
				else
				{
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 0;
					in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 0;
				}
			}
			else
			{
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = 0;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = 0;
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 0;
			}
		}
	}
	return true;
}

void ImageProcessing::DoG_SIFT(IMAGEINFOR in)
{
	int i, j, maxx, minn, orinm;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	//畫圓
	int cxp[] = { 0,1,2,2,2,1,0,-1,-2,-2,-2,-1 };
	int cyp[] = { -2,-2,-1,0,1,2,2,2,1,0,-1,-2 };

	IMAGEINFOR a, b, c;
	a.BmpInfo.width  = w;
	a.BmpInfo.height = h;
	a.BmpInfo.ImgData = new BYTE[w * h * 3];
	a.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(a.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	b.BmpInfo.width  = w;
	b.BmpInfo.height = h;
	b.BmpInfo.ImgData = new BYTE[w * h * 3];
	b.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(b.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	c.BmpInfo.width  = w;
	c.BmpInfo.height = h;
	c.BmpInfo.ImgData = new BYTE[w * h * 3];
	c.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(c.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	DoGFilter(a, 0.3, 0.4, 9);
	DoGFilter(b, 0.7, 0.8, 9);
	DoGFilter(c, 1.1, 1.2, 9);

	int krn = (3 - 1) / 2;
	for (int n = 1; n < h - 2; n++)
	{
		for (int m = 1; m < w - 2; m++)
		{
			maxx = minn = orinm = b.BmpInfo.ImgData[(n * w * 3) + (3 * m)];
			for (i = -krn; i < krn + 1; i++)
			{
				for (j = -krn; j < krn + 1; j++)
				{
					if (a.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] > maxx)
						maxx = a.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];

					if (a.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] < minn)
						minn = a.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];

					if (i != 0 && j != 0)	//中心No Run
					{
						if (b.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] > maxx)
							maxx = b.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];

						if (b.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] < minn)
							minn = b.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];
					}

					if (c.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] > maxx)
						maxx = c.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];

					if (c.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))] < minn)
						minn = c.BmpInfo.ImgData[((n + i) * w * 3) + (3 * (m + j))];
				}
			}

			if (orinm < 255 && orinm > 0)	//閥值
			{
				if (orinm == maxx)
				{
					for (i = 0; i < 12; i++)
					{
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i]))] = 0;
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 1] = 255;
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 2] = 30;
					}
				}
				else if (orinm == minn)
				{
					for (i = 0; i < 12; i++)
					{
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i]))] = 0;
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 1] = 0;
						in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 2] = 255;
					}
				}
			}
		}
	}
	delete[]a.BmpInfo.ImgData;
	delete[]b.BmpInfo.ImgData;
	delete[]c.BmpInfo.ImgData;
}

void ImageProcessing::FAST(IMAGEINFOR in, int jd)
{
	int i, j, mct = 0, nct = 0, thd = 10; //thd 閥值
	int w = ImagInfo.BmpInfo.width, h = ImagInfo.BmpInfo.height;

	IMAGEINFOR b;
	b.BmpInfo.ImgData = new BYTE[w * h * 3];
	b.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(b.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	Grayscale(b);

	int xp[] = { 0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1 };
	int yp[] = { -3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3 };

	//畫圓
	int cxp[] = { 0,1,2,2,2,1,0,-1,-2,-2,-2,-1 };
	int cyp[] = { -2,-2,-1,0,1,2,2,2,1,0,-1,-2 };

	for (int n = 3; n < h - 4; n++)
	{
		for (int m = 3; m < w - 4; m++)
		{
			for (i = 0; i < 16; i++)
			{
				if (b.BmpInfo.ImgData[((n + yp[i]) * w * 3) + (3 * (m + xp[i]))] < b.BmpInfo.ImgData[(n * w * 3) + (3 * m)] - thd)
					mct++;
				else
					mct = 0;

				if (b.BmpInfo.ImgData[((n + yp[i]) * w * 3) + (3 * (m + xp[i]))] > b.BmpInfo.ImgData[(n * w * 3) + (3 * m)] + thd)
					nct++;
				else
					nct = 0;
			}

			if (mct >= jd)
			{
				for (i = 0; i < 12; i++)
				{
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i]))] = 0;
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 1] = 255;
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 2] = 30;
				}
			}

			if (nct >= jd)
			{
				for (i = 0; i < 12; i++)
				{
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i]))] = 0;
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 1] = 0;
					in.BmpInfo.ImgData[((n + cyp[i]) * w * 3) + (3 * (m + cxp[i])) + 2] = 255;
				}
			}
			mct = 0;
			nct = 0;
		}
	}
	delete[]b.BmpInfo.ImgData;
}

void ImageProcessing::FAST_FAST(IMAGEINFOR in, int jd, int ctt)
{
	bool mflag = false;
	int n, m, i, j, k, rpe, ct = 0, thd = 30; //thd 閥值
	int w = in.BmpInfo.width, h = in.BmpInfo.height;

	IMAGEINFOR b;
	b.BmpInfo.ImgData = new BYTE[w * h * 3];
	b.BmpInfo.imagesize = in.BmpInfo.imagesize;
	memcpy(b.BmpInfo.ImgData, in.BmpInfo.ImgData, sizeof(BYTE) * w * h * 3);

	int* Rpear = new int[w * h];
	memset(Rpear, 0, sizeof(int) * w * h);

	Grayscale(b);

	//四角點
	int pxp[] = { 0,3,0,-3 };
	int pyp[] = { -3,0,3,0 };

	int xp[] = { 0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1 };
	int yp[] = { -3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3 };

	//畫圓
	int cxp[] = { 0,1,2,2,2,1,0,-1,-2,-2,-2,-1 };
	int cyp[] = { -2,-2,-1,0,1,2,2,2,1,0,-1,-2 };

	for (n = 3; n < h - 4; n++)
	{
		for (m = 3; m < w - 4; m++)
		{
			for (i = 0; i < 4; i++)
			{
				if (abs(b.BmpInfo.ImgData[((n + pyp[i]) * w * 3) + (3 * (m + pxp[i]))] - b.BmpInfo.ImgData[(n * w * 3) + (3 * m)]) > thd)
					ct++;
			}

			if (ct >= ctt)
			{
				ct = 0;
				for (i = 0; i < 16; i++)
				{
					int da = abs(b.BmpInfo.ImgData[((n + yp[i]) * w * 3) + (3 * (m + xp[i]))] - b.BmpInfo.ImgData[(n * w * 3) + (3 * m)]);
					rpe += da;
					if (da > thd)
						ct++;
					else
						ct = 0;
				}

				if (ct >= jd)
					Rpear[(n * w) + m] = rpe;
				rpe = 0;
			}
			ct = 0;
		}
	}

	//非極大值抑制
	for (n = 3; n < h - 4; n++)
	{
		for (m = 3; m < w - 4; m++)
		{
			if (Rpear[(n * w) + m] > 0)
			{
				for (i = -2; i < 3; i++)
				{
					for (j = -2; j < 3; j++)
					{
						if (Rpear[((n + i) * w) + (m + j)] > Rpear[(n * w) + m])
						{
							mflag = true;
							break;
						}
					}
				}

				if (mflag == false)
				{
					for (k = 0; k < 12; k++)
					{
						in.BmpInfo.ImgData[((n + cyp[k]) * w * 3) + (3 * (m + cxp[k]))] = 0;
						in.BmpInfo.ImgData[((n + cyp[k]) * w * 3) + (3 * (m + cxp[k])) + 1] = 242;
						in.BmpInfo.ImgData[((n + cyp[k]) * w * 3) + (3 * (m + cxp[k])) + 2] = 255;
					}
				}
				mflag = false;
			}
		}
	}
	delete[]b.BmpInfo.ImgData;
	delete[]Rpear;
}

bool ImageProcessing::ImageFeatureDetection(IMAGEINFOR in, int mod)	//特徵偵測
{
	if (in.BmpInfo.ImgData == NULL) return false;
	switch (mod)
	{
	case ImgFD_SIFT:
		DoG_SIFT(in);
		break;
	case ImgFD_FAST_9:
		FAST(in, 9);
		break;
	case ImgFD_FAST_12:
		FAST(in, 12);
		break;
	case ImgFD_FAST_FAST_9:
		FAST_FAST(in, 9, 2);
		break;
	case ImgFD_FAST_FAST_12:
		FAST_FAST(in, 12, 3);
		break;
	}
	return true;
}

bool ImageProcessing::CNN_()
{
	Minst minst;
	CNNNet cnnnet;
	//cnnnet.est_minst();	//OK
	//cnnnet.test_mat1();	//OK
	//cnnnet.test_cnn();	//OK

	LabelArr trainLabel = minst.read_Lable("C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\train-labels.idx1-ubyte");
	ImgArr trainImg     = minst.read_Img("C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\train-images.idx3-ubyte");
	LabelArr testLabel  = minst.read_Lable("C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\test-labels.idx1-ubyte");
	ImgArr testImg      = minst.read_Img("C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\test-images.idx3-ubyte");

	nSize inputSize = { testImg->ImgPtr[0].c,testImg->ImgPtr[0].r };
	int outSize = testLabel->LabelPtr[0].l;

	//CNN结构的初始化
	CNN* cnn = (CNN*)malloc(sizeof(CNN));
	cnnnet.cnnsetup(cnn, inputSize, outSize);

	//CNN训练
	CNNOpts opts;
	opts.numepochs=1;
	opts.alpha=1.0;
	int trainNum=55000;
	cnnnet.cnntrain(cnn,trainImg,trainLabel,opts,trainNum);
	printf("train finished!!\n");
	cnnnet.savecnn(cnn,"C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\Data\\minst.cnn");

	// 保存训练误差
	FILE  *fp = NULL;
	fp=fopen("C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\cnnL.ma","wb");
	if(fp == NULL)
		printf("write file failed\n");
	fwrite(cnn->L,sizeof(float),trainNum,fp);
	fclose(fp);

	//CNN测试
	cnnnet.importcnn(cnn, "C:\\Users\\CWChang\\Desktop\\Img處理\\CNN\\Data\\minst.cnn");
	int testNum = 10000;
	float incorrectRatio = 0.0;
	incorrectRatio = cnnnet.cnntest(cnn, testImg, testLabel, testNum);
	printf("incorrectRatio = %.3f\n", incorrectRatio);
	printf("test finished!!\n");
	return true;
}

bool ImageProcessing::DNN_()
{
	for (rint i = kn, j, LX, LY; i; i--)
		for (LX = rd() % RANGE + 1, LY = rd() % RANGE + 1, j = rd() % (NUM_PER_SET_MAX - NUM_PER_SET_MIN) + NUM_PER_SET_MIN; j--;)
			Set[i].push_back(P{ rd() % SET_RANGE + LX, rd() % SET_RANGE + LY, i - 1 });

	for (rint i, j; T--; printf("%.6lf %.6lf\n", Loss / Exam.size(), Score / Exam.size()))
	{
		for (vector<P>().swap(Homework), vector<P>().swap(Exam), i = 1; i <= kn; i++)
			for (shuffle(Set[i].begin(), Set[i].end(), rd), j = 0; j < Set[i].size(); j++)
				(j < NUM_PER_SET_MIN - 10 ? Homework : Exam).push_back(Set[i][j]);

		NeuralNetwork NN({ 2, 50, kn }, .1);
		Loss = Score = 0;
		shuffle(Homework.begin(), Homework.end(), rd);
		_ = 0;

		for (auto p : Homework)
		{
			NN.In = { (double)p.x / RANGE, (double)p.y / RANGE };
			NN.Ans = Arr(kn);
			NN.Ans[p.i] = 1;
			NN.Run();		//正向
			NN.Adjust();	//反向
			/*
			printf("Training #%d:\n",++_);
			for(j = 0; j < K; printf("%4.4lf ",NN.Out[j]), j++); puts("");
			for(j = 0; j < K; printf("%4.4lf ",NN.Ans[j]), j++); puts("");
			printf("Output = %d, Answer = %d\n\n",NN.Judge(),p.i);
			//*/
		}

		shuffle(Exam.begin(), Exam.end(), rd);
		_ = 0;

		for (auto p : Exam)
		{
			NN.In = { (double)p.x / RANGE, (double)p.y / RANGE };
			NN.Ans = Arr(kn);
			NN.Ans[p.i] = 1;
			NN.Run();
			Loss += NN.CELoss();
			Score += NN.Judge() == p.i;
			/*
			printf("Testing #%d:\n",++_);
			for(j = 0; j < K; printf("%4.4lf ",NN.Out[j]), j++); puts("");
			for(j = 0; j < K; printf("%4.4lf ",NN.Ans[j]), j++); puts("");
			printf("Output = %d, Answer = %d\n\n",NN.Judge(),p.i);
			//*/
		}
	}
	return true;
}

void ImageProcessing::YoloV3()
{
	Modelstr str;
	ImgObjDetection ImgOjd;
	str.classesFile		   = "D:\\Yolo\\coco.names";
	str.modelConfiguration = "D:\\Yolo\\yolov3.cfg";
	str.modelWeights	   = "D:\\Yolo\\yolov3.weights";
	ImgOjd.DetectStreaming(str);
}

void ImageProcessing::YoloV3(IMAGEINFOR in)
{
	Modelstr str;
	ImgObjDetection ImgOjd;
	str.classesFile		   = "D:\\Yolo\\coco.names";
	str.modelConfiguration = "D:\\Yolo\\yolov3.cfg";
	str.modelWeights	   = "D:\\Yolo\\yolov3.weights";
	ImgOjd.DetectImage(in.BmpInfo.ImgData, in.BmpInfo.width, in.BmpInfo.height, str);
}

void ImageProcessing::YoloV3(const char* vstr, const char* ostr)
{
	Modelstr str;
	ImgObjDetection ImgOjd;
	str.classesFile		   = "D:\\Yolo\\coco.names";
	str.modelConfiguration = "D:\\Yolo\\yolov3.cfg";
	str.modelWeights	   = "D:\\Yolo\\yolov3.weights";
	ImgOjd.DetectVideo(vstr, ostr,str);
}

bool ImageProcessing::ImageSemanticSegmentation()
{
	
	return true;
}

bool ImageProcessing::VideoObjectTracking()
{

	return true;
}

bool ImageProcessing::Threshold(IMAGEINFOR in, int thd)
{
	if (in.BmpInfo.ImgData == NULL) return false;
	int w = in.BmpInfo.width, h = in.BmpInfo.height;
	for (int n = 0; n < h; n++)
	{
		for (int m = 0; m < w; m++)
		{
			if (in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] > thd)
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 255;
			if (in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] < thd)
				in.BmpInfo.ImgData[(n * w * 3) + (3 * m)] = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 1] = in.BmpInfo.ImgData[(n * w * 3) + (3 * m) + 2] = 0;
		}
	}
	return true;
}