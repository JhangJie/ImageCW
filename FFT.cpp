#include "pch.h"
#include "FFT.h"
#include "math.h"
#include "ImgMath.h"
#include <iostream> 

#define M_PI 3.1415926

Complex::Complex()
{

}

Complex::~Complex()
{

}

double Complex::Real()
{
	return real;
}

double Complex::Imaginary()
{
	return imaginary;
}

void Complex::Real_(double v)
{
	real = v;
}

void Complex::Imaginary_(double v)
{
	imaginary = v;
}

Complex Complex::Complex_(double dl, double dg)
{
	Complex c;
	c.real = dl;
	c.imaginary = dg;
	return c;
}

Complex Complex::Complex_(Complex o)
{
	Complex c;
	c.real = o.real;
	c.imaginary = o.imaginary;
	return c;
}

Complex  Complex::add(Complex c1, Complex c2)
{
	return c1.cadd(c2);
}

Complex  Complex::sub(Complex c1, Complex c2)
{
	return c1.csub(c2);
}

Complex  Complex::mul(Complex c1, Complex c2)
{
	return c1.cmul(c2);
}

Complex  Complex::cadd(Complex cp)
{
	double x = real + cp.real;
	double y = imaginary + cp.imaginary;
	return Complex_(x, y);
}

Complex  Complex::csub(Complex cp)
{
	double x = real - cp.real;
	double y = imaginary - cp.imaginary;
	return Complex_(x, y);
}

Complex  Complex::cmul(Complex cp)
{
	double x = real * cp.real - imaginary * cp.imaginary;
	double y = real * cp.imaginary + imaginary * cp.real;
	return Complex_(x, y);
}

double Complex::Abs()	//幅值
{
	double x = abs((int)real);
	double y = abs((int)imaginary);
	if (real == 0) return y;
	if (imaginary == 0) return x;
	if (x > y)
		return (x * sqrt(1 + (y / x) * (y / x)));
	else
		return (y * sqrt(1 + (x / y) * (x / y)));
}

double Complex::Angle()	//相位角
{
	if (real == 0 && imaginary == 0) return 0;

	if (real == 0)
	{
		if (imaginary > 0)
			return M_PI / 2;
		else
			return -M_PI / 2;
	}
	else
	{
		if (real > 0)
			return atan2(imaginary, real);
		else
		{
			if (imaginary >= 0)
				return atan2(imaginary, real) + M_PI;
			else
				return atan2(imaginary, real) - M_PI;
		}
	}
}

Complex Complex::conjugate() //共軛
{
	return Complex_(this->real, -(this->imaginary));
}

//DFT
DFT::DFT(unsigned char* inptr, int iw, int ih)
{
	ImgData = new BYTE[iw * ih * 3];
	memcpy(ImgData, inptr, sizeof(BYTE) * iw * ih * 3);
	w = iw;
	h = ih;
}

DFT::~DFT()
{

}

Complex* DFT::FFT(Complex* srcdata, int ctn)
{
	//求fft的級數
	Imgmath imgmath;
	int r = imgmath.Logbase(ctn, 2);	//影響img輸出
	int i, j, k;

	//求取修正係數W
	Complex c1;
	Complex* w = new Complex[ctn / 2];
	Complex* intvar1 = new Complex[ctn];
	Complex* intvar2 = new Complex[ctn];
	memcpy(intvar1, srcdata, sizeof(Complex) * ctn);

	double angle = 0;
	for (int i = 0; i < ctn / 2; i++)
	{
		angle = -i * M_PI * 2 / ctn;
		w[i] = c1.Complex_(cos(angle), sin(angle));
		angle = 0;
	}
	int intval = 0;
	int halfN = 0;
	int gap = 0;

	Complex t1, t2;

	//核心部分
	for (i = 0; i < r; i++)
	{
		intval = 1 << i;
		halfN = 1 << (r - i);
		for (j = 0; j < intval; j++)
		{
			gap = j * halfN;
			for (k = 0; k < halfN / 2; k++)
			{
				intvar2[k + gap] = t1.add(intvar1[k + gap], intvar1[k + gap + halfN / 2]);
				intvar2[k + halfN / 2 + gap] = t2.mul(t1.sub(intvar1[k + gap], intvar1[k + gap + halfN / 2]), w[k * intval]);
			}
		}
		memcpy(intvar1, intvar2, sizeof(Complex) * ctn);
	}
	unsigned int rev = 0;
	unsigned int num = 0;
	for (j = 0; j < ctn; j++)
	{
		rev = 0;
		num = j;
		for (i = 0; i < r; i++)
		{
			rev <<= 1;
			rev |= num & 1;
			num >>= 1;
		}
		intvar2[rev] = intvar1[j];
	}
	return intvar2;
}

Complex* DFT::FFT2(unsigned char* imgdata, bool inv)
{
	int i, j;
	BYTE* bmpv = new BYTE[w * h];
	memcpy(bmpv, imgdata, sizeof(BYTE) * w * h);

	Complex t1;
	Complex* tempc1 = new Complex[w * h];

	for (i = 0; i < w * h; i++)
	{
		if (inv == true)
		{
			if ((i / w + i % w) % 2 == 0)
				tempc1[i] = t1.Complex_(bmpv[i], 0);
			else
				tempc1[i] = t1.Complex_(-bmpv[i], 0);
		}
		else
		{
			tempc1[i] = t1.Complex_(bmpv[i], 0);
		}
	}

	Complex* tempc2 = new Complex[w];
	Complex* tempc3 = new Complex[w];
	for (i = 0; i < h; i++)	//水平方向
	{
		for (j = 0; j < w; j++)
			tempc2[j] = tempc1[i * w + j];
		tempc3 = FFT(tempc2, w);
		for (j = 0; j < w; j++)
			tempc1[i * w + j] = tempc3[j];
	}

	Complex* tempc4 = new Complex[h];
	Complex* tempc5 = new Complex[h];
	for (i = 0; i < w; i++)	//垂直方向
	{
		for (j = 0; j < h; j++)
			tempc4[j] = tempc1[j * w + i];
		tempc5 = FFT(tempc4, h);
		for (j = 0; j < h; j++)
			tempc1[j * w + i] = tempc5[j];
	}
	return tempc1;
}

BYTE* DFT::FFTImg()
{
	int i, n, m;
	Imgmath imgmath;
	BYTE* gbuf = new BYTE[w * h * 3];
	memcpy(gbuf, ImgData, sizeof(BYTE) * w * h * 3);
	ImgData_ = new BYTE[w * h];

	for (n = 0; n < h; n++)
		for (m = 0; m < w; m++)
			ImgData_[n * w + m] = gbuf[n * w * 3 + 3 * m];

	Complex* frdm = new Complex[w * h];
	double* tmparr = new double[w * h];
	frdm = FFT2(ImgData_, true);
	for (n = 0; n < w * h; n++)
		tmparr[n] = imgmath.Logbase(1 + frdm[n].Abs(), 2);

	double a = 1000.0, b = 0.0;
	double p;
	for (i = 0; i < w * h; i++)
	{
		if (a > tmparr[i])
			a = tmparr[i];
		if (b < tmparr[i])
			b = tmparr[i];
	}
	p = 255.0 / (b - a);
	for (i = 0; i < w * h; i++)
		ImgData_[i] = (BYTE)(p * (tmparr[i] - a) + 0.5);

	BYTE* dstbuf = new BYTE[w * h * 3];
	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			dstbuf[(n * w * 3) + (3 * m)] = ImgData_[n * w + m];
			dstbuf[(n * w * 3) + (3 * m) + 1] = ImgData_[n * w + m];
			dstbuf[(n * w * 3) + (3 * m) + 2] = ImgData_[n * w + m];
		}
	}
	memcpy(gbuf, dstbuf, sizeof(BYTE) * w * h * 3);
	return gbuf;
}

BYTE* DFT::FFTPhaseImg()
{
	int i, n, m;
	BYTE* gbuf = new BYTE[w * h * 3];
	memcpy(gbuf, ImgData, sizeof(BYTE) * w * h * 3);
	ImgData_ = new BYTE[w * h];

	for (n = 0; n < h; n++)
		for (m = 0; m < w; m++)
			ImgData_[n * w + m] = gbuf[n * w * 3 + 3 * m];

	Complex* frdm = new Complex[w * h];
	double* tmparr = new double[w * h];
	frdm = FFT2(ImgData_, true);
	for (n = 0; n < w * h; n++)
		tmparr[n] = frdm[n].Angle() + 2 * M_PI;

	double a = 1000.0, b = 0.0;
	double p;
	for (i = 0; i < w * h; i++)
	{
		if (a > tmparr[i])
			a = tmparr[i];
		if (b < tmparr[i])
			b = tmparr[i];
	}
	p = 255.0 / (b - a);
	for (i = 0; i < w * h; i++)
		ImgData_[i] = (BYTE)(p * (tmparr[i] - a) + 0.5);

	BYTE* dstbuf = new BYTE[w * h * 3];
	for (n = 0; n < h; n++)
	{
		for (m = 0; m < w; m++)
		{
			dstbuf[(n * w * 3) + (3 * m)] = ImgData_[n * w + m];
			dstbuf[(n * w * 3) + (3 * m) + 1] = ImgData_[n * w + m];
			dstbuf[(n * w * 3) + (3 * m) + 2] = ImgData_[n * w + m];
		}
	}
	memcpy(gbuf, dstbuf, sizeof(BYTE) * w * h * 3);
	return gbuf;
}






