#include <windows.h>

class Complex
{
private:
	double real;
	double imaginary;

public:
	Complex();
	~Complex();

	double Real();
	double Imaginary();
	void Real_(double v);
	void Imaginary_(double v);

	Complex Complex_(double dl, double dg);
	Complex Complex_(Complex o);

	Complex  add(Complex c1, Complex c2);
	Complex  sub(Complex c1, Complex c2);
	Complex  mul(Complex c1, Complex c2);

	Complex  cadd(Complex cp);
	Complex  csub(Complex cp);
	Complex  cmul(Complex cp);

	double Abs();
	double Angle();

	Complex conjugate();
};

class DFT
{
private:
	BYTE* ImgData;
	BYTE* ImgData_;
	int w;
	int h;

	Complex* FFT(Complex* srcdata, int ctn);			//一維FFT變換
	Complex* FFT2(unsigned char* imgdata, bool inv);	//二維FFT變換

public:
	DFT(unsigned char* inptr, int iw, int ih);
	~DFT();

	BYTE* FFTImg();
	BYTE* FFTPhaseImg();
};


