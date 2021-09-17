#include "math.h"

class Imgmath
{
private:


public:
	Imgmath();
	~Imgmath();

	bool Mean(unsigned char* inptr, double* bMean);
	bool MaxMin(unsigned char* inptr, double* Max, double* Min);
	bool STDEV(unsigned char* inptr, double* Stdev, double* Mean);
	bool F_STDEV(float* inptr, double* Stdev, double* Mean);
	double Logbase(double a,double base);
};