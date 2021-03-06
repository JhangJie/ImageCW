#include "pch.h"
#include "ImgProcessing.h"

Imagematting::Imagematting()
{
	bsize = 0;
	fsize = 0;
	usize = 0;
	covarienceOfMat = new float[3 * 3];
	avgOfMat = new float[1 * 3];
}

Imagematting::~Imagematting()
{
	delete[]img;
	delete[]trimap;
	delete[]matte;
	delete[]dilateimg;
	delete[]covarienceOfMat;
	delete[]avgOfMat;
}

int AT_(float* mat, int x, int y)	//?ݭץ?
{
	return mat[x * y];
}

void AT(float* mat, int x, int y, int data)	//?ݭץ?
{
	mat[x * y] = data;
}

void Imagematting::dilate(unsigned char* inptr, int w, int h)
{
	//ImageProcessing imgPr;
	//imgPr.Dilation(imgPr.ImagInfo, w, h);
	//dilateimg = inptr;
}

void Imagematting::addInMat(float* mat, int n, int i, int j, int b, int g, int r)
{
	AT(mat, n, 0, i);
	AT(mat, n, 1, j);
	AT(mat, n, 2, b);
	AT(mat, n, 3, g);
	AT(mat, n, 4, r);
}

void Imagematting::addInMat(float* mat, int n, int x, int y)
{
	AT(mat, n, 0, x);
	AT(mat, n, 1, y);
}

void Imagematting::createMat()
{
	BYTE* udata = trimap;
	BYTE* ddata = dilateimg;

	int bn = 0, fn = 0, un = 0, n = 0;

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			int bc = data[i * step + j * channels];
			int gc = data[i * step + j * channels + 1];
			int rc = data[i * step + j * channels + 2];
			int gray = udata[i * g_step + j];
			int dgray = ddata[i * g_step + j];

			if (gray < 5) // gray is background (gray == 4!!!!!!!!!!!!!!)
			{
				// add (x,y,r,g,b) to allmat
				addInMat(bmat, bn, i, j);
				bn++;
			}
			else if (gray > 250) // gray is foreground (gray == 252!!!!!!!!!!)
			{
				// add (x,y,r,g,b) to allmat
				addInMat(fmat, fn, i, j);
				fn++;
			}
			else // otherwise
			{
				// add (x,y,r,g,b) to allmat
				addInMat(umat, un, i, j);
				un++;
			}
			if (dgray == 255)
			{
				addInMat(allmat, n, i, j, bc, gc, rc);
				// xy_index[i][j] = n;
				n++;
			}
		}
	}
	printf("bn = %d , fn = %d , un = %d , n = %d\n", bn, fn, un, n);
	printf("createmat ok \n");
}

void Imagematting::loadImage(unsigned char* inptr, int w, int h)
{
	CMat mat;
	height = h;
	width = w;
	step = w * 3;
	channels = 3;
	data = inptr;

	tri = new int* [height];
	preAlpha = new double* [height];
	confidence = new double* [height];
	preAlpha = new double* [height];
	confidence = new double* [height];

	for (int i = 0; i < height; ++i)
	{
		tri[i] = new int[width];
		preAlpha[i] = new double[width];
		confidence[i] = new double[width];
		//xy_index[i] = new int[width];
	}

	N = height * width + 2;
	printf("N = %d\n", N);

	W1 = mat.matrix_new(N, N);	//?}???x?}
	W2 = mat.matrix_new(N, N);
	W3 = mat.matrix_new(N, N);
	L  = mat.matrix_new(N, N);
	I  = mat.matrix_new(N, N);

	Alpha = new unsigned char[N];
	memset(Alpha, 1, sizeof(unsigned char) * N);
	G = new unsigned char[N];
	memset(G, 1, sizeof(unsigned char) * N);

	matte = new unsigned char[w * h * 3];	//?`?N
	memset(matte, 0, sizeof(unsigned char) * w * h * 3);
	dilateimg = new unsigned char[w * h * 3];
	memset(dilateimg, 0, sizeof(unsigned char) * w * h * 3);
}

void Imagematting::loadTrimap(unsigned char* inptr, int w, int h)
{
	trimap = inptr;
	g_step = w * 3;

	BYTE* binaryimg = new BYTE[w * h * 3];
	memcpy(binaryimg, inptr, sizeof(BYTE) * w * h * 3);
	//memset(binaryimg,0,sizeof(unsigned char)*w*h*3);

	BYTE* bdata = binaryimg;

	if (trimap == NULL) return;

	BYTE* udata = trimap;
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			int gray = udata[i * g_step + j];
			if (gray < 5) // c is a background
			{
				bsize++;
				tri[i][j] = 0;
				preAlpha[i][j] = 0;
				confidence[i][j] = 0; // only unknown pixels have confidence
				bdata[i * g_step + j] = 0;
			}
			else if (gray > 250) // c is a foreground
			{
				fsize++;
				tri[i][j] = 1;
				preAlpha[i][j] = 1;
				confidence[i][j] = 0;
				bdata[i * g_step + j] = 0;
			}
			else // c is a unknown pixel
			{
				usize++;
				tri[i][j] = 2;
				bdata[i * g_step + j] = 255;
			}
		}
	}

	//dilate(binaryimg, w, h);
	//delete[]binaryimg;
	dilate(dilateimg, w, h);

	allsize = 0;
	for (int n = 0; n < h; n++)	//?p??Pixel??0 Count ( ?G?Ȥ?data )
	{
		for (int m = 0; m < w; m++)
		{
			//printf("dilateimg = %d\n",dilateimg[n * w * 3 + 3 * m]);
			if (dilateimg[n * w * 3 + 3 * m] == 0) {
				allsize++;
				//printf("allsize = %d\n",allsize);
			}
		}
	}
	printf("allsize = %d\n", allsize);

	bmat = new float[bsize * 2];
	memset(bmat, 0, sizeof(float) * bsize * 2);

	fmat = new float[fsize * 2];
	memset(fmat, 0, sizeof(float) * fsize * 2);

	umat = new float[usize * 2];
	memset(umat, 0, sizeof(float) * usize * 2);

	allmat = new float[allsize * 2];
	memset(allmat, 0, sizeof(float) * allsize * 2);

	createMat();
	printf("loadtrimap ok \n");
}

void Tree_knnSearch(unsigned char* inptr)	//search kd-tree
{


}

void Imagematting::findKnearest()			//build 2 KD-trees
{




	printf("get kdtree ok \n");
}

int  Imagematting::BC(float* mat, int index)
{
	//return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels];
	return 0;
}

int  Imagematting::GC(float* mat, int index)
{
	//return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels + 1];
	return 0;
}

int  Imagematting::RC(float* mat, int index)
{
	//return data[mat.at<int>(index, 0) * step + mat.at<int>(index, 1) * channels + 2];
	return 0;
}

double Imagematting::geteveryAlpha(int c, int f, int b) //f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	int findex = AT_(fresult.indices, c, f);
	int bindex = AT_(bresult.indices, c, b);

	double alpha = ((BC(umat, c) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
		(GC(umat, c) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
		(RC(umat, c) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex)))
		/ ((BC(fmat, findex) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
			(GC(fmat, findex) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
			(RC(fmat, findex) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex)) + 0.0000001);
	return MIN(1.0, MAX(0.0, alpha));
}

double Imagematting::getRd(int c, int f, int b)			//f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	double alpha = geteveryAlpha(c, f, b);
	int findex = AT_(fresult.indices, c, f);
	int bindex = AT_(bresult.indices, c, b);

	double result = sqrt(((BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) * (BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) +
		(GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) * (GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) +
		(RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex)) * (RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex))) /
		(((BC(fmat, findex) - BC(bmat, bindex)) * (BC(fmat, findex) - BC(bmat, bindex)) +
			(GC(fmat, findex) - GC(bmat, bindex)) * (GC(fmat, findex) - GC(bmat, bindex)) +
			(RC(fmat, findex) - RC(bmat, bindex)) * (RC(fmat, findex) - RC(bmat, bindex))) + 0.0000001));
	return result;
}

void Imagematting::getD() // correspond to umat
{
	// get db, df of every pixel
	dB = new int[usize];
	dF = new int[usize];
	double min1 = 0, min2 = 0;
	for (int i = 0; i < usize; i++)
	{
		// calculate d2
		dB[i] = 200000; // need to more than 255^2 * 3
		dF[i] = 200000;
		for (int k = 0; k < K; k++)
		{
			int bindex = AT_(bresult.indices, i, k); // get the nearest background of C
			min1 = (BC(umat, i) - BC(bmat, bindex)) * (BC(umat, i) - BC(bmat, bindex)) + (GC(umat, i) - GC(bmat, bindex)) * (GC(umat, i) - GC(bmat, bindex)) + (RC(umat, i) - RC(bmat, bindex)) * (RC(umat, i) - RC(bmat, bindex));
			if (min1 < dB[i]) dB[i] = min1;

			int findex = AT_(fresult.indices, i, k); // get the nearest foreground of C
			min2 = (BC(umat, i) - BC(fmat, findex)) * (BC(umat, i) - BC(fmat, findex)) + (GC(umat, i) - GC(fmat, findex)) * (GC(umat, i) - GC(fmat, findex)) + (RC(umat, i) - RC(fmat, findex)) * (RC(umat, i) - RC(fmat, findex));
			if (min2 < dF[i]) dF[i] = min2;
		}
	}
	printf("getD ok \n");
}

double Imagematting::getW(int c, int fb, bool flag) // flag == 1, f; flag == 0, b; fb is the fbth-nearest fore- or background pixel of C
{
	double w;

	if (flag == 0) // b
	{
		int index = AT_(bresult.indices, c, fb);
		w = 1 - exp(-((BC(umat, c) - BC(bmat, index)) * (BC(umat, c) - BC(bmat, index)) +
			(GC(umat, c) - GC(bmat, index)) * (GC(umat, c) - GC(bmat, index)) +
			(RC(umat, c) - RC(bmat, index)) * (RC(umat, c) - RC(bmat, index))) / (dB[c] + 0.0000001));
	}
	else // f
	{
		int index = AT_(fresult.indices, c, fb);
		w = 1 - exp(-((BC(umat, c) - BC(fmat, index)) * (BC(umat, c) - BC(fmat, index)) +
			(GC(umat, c) - GC(fmat, index)) * (GC(umat, c) - GC(fmat, index)) +
			(RC(umat, c) - RC(fmat, index)) * (RC(umat, c) - RC(fmat, index))) / (dF[c] + 0.0000001));
	}
	return w;
}

double Imagematting::getEc(int c, int f, int b)		//f is the fth-nearest pixel of C, b is the bth-nearest pixel of C
{
	double alpha = geteveryAlpha(c, f, b);
	int findex = AT_(fresult.indices, c, f);
	int bindex = AT_(bresult.indices, c, b);

	double result = sqrt(((BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) * (BC(umat, c) - alpha * BC(fmat, findex) - (1 - alpha) * BC(bmat, bindex)) +
		(GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) * (GC(umat, c) - alpha * GC(fmat, findex) - (1 - alpha) * GC(bmat, bindex)) +
		(RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex)) * (RC(umat, c) - alpha * RC(fmat, findex) - (1 - alpha) * RC(bmat, bindex))));
	return result;
}

double Imagematting::getEs(int c, int fb, bool flag) // flag == 1, f; flag == 0, b; fb is the fbth-nearest fore- or background pixel of C
{
	double w;
	if (flag == 1) // f
	{
		int index = AT_(fresult.indices, c, fb);
		w = sqrt((AT_(umat, c, 0) - AT_(fmat, index, 0)) * (AT_(umat, c, 0) - AT_(fmat, index, 0)) +
			(AT_(umat, c, 1) - AT_(fmat, index, 1)) * (AT_(umat, c, 1) - AT_(fmat, index, 1)) / (dsF[c] + 0.0000001));
	}
	else // b
	{
		int index = AT_(bresult.indices, c, fb);
		w = sqrt((AT_(umat, c, 0) - AT_(bmat, index, 0)) * (AT_(umat, c, 0) - AT_(bmat, index, 0)) +
			(AT_(umat, c, 1) - AT_(bmat, index, 1)) * (AT_(umat, c, 1) - AT_(bmat, index, 1)) / (dsB[c] + 0.0000001));
	}
	return w;
}

void Imagematting::getDspace() // not sqrt
{
	// get db, df of every pixel
	dsB = new int[usize];
	dsF = new int[usize];
	double min1 = 0, min2 = 0;
	for (int i = 0; i < usize; i++)
	{
		// calculate d2
		dsB[i] = height * height + width * width;	 // need to more than height^2 + width^2
		dsF[i] = height * height + width * width;
		for (int k = 0; k < K; k++)
		{
			int bindex = AT_(bresult.indices, i, k); // get the nearest background of C
			min1 = (AT_(umat, i, 0) - AT_(bmat, bindex, 0)) * (AT_(umat, i, 0) - AT_(bmat, bindex, 0)) + (AT_(umat, i, 1) - AT_(bmat, bindex, 1)) * (AT_(umat, i, 1) - AT_(bmat, bindex, 1));
			if (min1 < dsB[i]) dsB[i] = min1;

			int findex = AT_(fresult.indices, i, k); // get the nearest foreground of C
			min2 = (AT_(umat, i, 0) - AT_(fmat, findex, 0)) * (AT_(umat, i, 0) - AT_(fmat, findex, 0)) + (AT_(umat, i, 1) - AT_(fmat, findex, 1)) * (AT_(umat, i, 1) - AT_(fmat, findex, 1));
			if (min2 < dsF[i]) dsF[i] = min2;
		}
	}
	printf("getD ok \n");
}

double Imagematting::getConfidence(int c, int f, int b)  //f is the fth-nearest foreground pixel of C, b is the bth-nearest background pixel of C
{
	double confi;
	//confi = exp(-(getRd(c, f, b) * getRd(c, f, b) * getW(c, f, 1) * getW(c, b, 0)) / (sigma * sigma));
	confi = 1.0 / (getEc(c, f, b) + getEs(c, f, 1) + getEs(c, b, 0));
	return confi;
}

void Imagematting::getPreAlpha()
{
	//getD();
	getDspace();

	// calculate confidence of every unknown pixel
	for (int i = 0; i < usize; i++)
	{
		int Ci = AT_(umat, i, 0);
		int Cj = AT_(umat, i, 1);
		// choose three pairs which have the biggest confidence of every unknown pixel and mean their alphas as the predicted alpha --- fAlpha
		double alpha1 = 0, alpha2 = 0, alpha3 = 0;
		double confi1 = 0, confi2 = 0, confi3 = 0; // 1 > 2 > 3
		for (int f = 0; f < K; f++)
		{
			for (int b = 0; b < K; b++)
			{
				double confi = getConfidence(i, f, b);
				if (confi > confi1)
				{
					alpha3 = alpha2; confi3 = confi2;
					alpha2 = alpha1; confi2 = confi1;
					alpha1 = geteveryAlpha(i, f, b); confi1 = confi;
				}
				else if (confi < confi1 && confi > confi2)
				{
					alpha3 = alpha2; confi3 = confi2;
					alpha2 = geteveryAlpha(i, f, b); confi2 = confi;
				}
				else if (confi < confi2 && confi > confi3)
				{
					alpha3 = geteveryAlpha(i, f, b); confi3 = confi;
				}
			}
		}
		// get preAlpha and confidence
		preAlpha[Ci][Cj] = (alpha1 + alpha2 + alpha3) / 3.0;
		confidence[Ci][Cj] = (confi1 + confi2 + confi3) / 3.0;
	}

	// save in ".txt"
	// fstream f("a2.txt", ios::out);
	// if (!f) cout << "Error!" << endl;
	// for (int i = 0; i < height; i++)
	//   {
	//     for (int j = 0; j < width; j++)
	//       {
	//         f << preAlpha[i][j] << endl;
	//       }
	//   }
	// f.close();
	printf("getPreAlpha ok \n");
}

/*void Imagematting::TEST(SpMat A) // print all non-zero elements in Matrix A (just for test)
{
	for (int k = 0; k < A.outerSize(); ++k)
	{
		for (SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
		{
			cout << it.value() << "   ";
		}
	}
}*/

void   Imagematting::getWeight1() // get data term W(i, F) & W(i, B)
{
	/*std::vector<T> triplets;
	for (int i = 0; i < allsize; i++) // 0, 1 are two virtue nodes
	{
		// int x = i / width;
		//int y = i - x * width;
		int x = AT_(allmat, i, 0);
		int y = AT_(allmat, i, 1);
		int index = x * width + y + 2; // index in N
		if (tri[x][y] == 0 || tri[x][y] == 1) // known pixels
		{
			triplets.push_back(T(index, 0, -gamma * tri[x][y]));
			triplets.push_back(T(index, 1, -gamma * (1 - tri[x][y])));
			triplets.push_back(T(index, index, gamma)); // add to L(i, i)

			// triplets.push_back(T(0, index, -gamma * tri[x][y]));
			// triplets.push_back(T(1, index, -gamma * (1 - tri[x][y])));
			// triplets.push_back(T(0, 0, gamma * tri[x][y]));
			// triplets.push_back(T(1, 1, gamma * (1 - tri[x][y])));
		}
		else // unknown pixels
		{
			triplets.push_back(T(index, 0, -gamma * preAlpha[x][y]));
			triplets.push_back(T(index, 1, -gamma * (1 - preAlpha[x][y])));
			triplets.push_back(T(index, index, gamma)); // add to L(i, i)

			// triplets.push_back(T(0, index, -gamma * preAlpha[x][y]));
			// triplets.push_back(T(1, index, -gamma * (1 - preAlpha[x][y])));
			// triplets.push_back(T(0, 0, gamma * preAlpha[x][y]));
			// triplets.push_back(T(1, 1, gamma * (1 - preAlpha[x][y])));
		}
	}
	W1.setFromTriplets(triplets.begin(), triplets.end());
	W1.prune(0.0);*/
	printf("getWeight1 ok\n");
}

void   Imagematting::getCovarianceMatrix(int x, int y) // (x, y) are the up-left points in one 3*3 window (require: not the edge)
{
	int M = winSize * winSize; // the number of pixel in the window
	int n = 3; // channels
	/*CvMat* mat = cvCreateMat(M, n, CV_64FC1);

	// set the original mat
	for (int i = 0; i < winSize; i++)
	{
		for (int j = 0; j < winSize; j++)
		{
			cvmSet(mat, i * winSize + j, 0, data[(x + i) * step + (y + j) * channels]);
			cvmSet(mat, i * winSize + j, 1, data[(x + i) * step + (y + j) * channels + 1]);
			cvmSet(mat, i * winSize + j, 2, data[(x + i) * step + (y + j) * channels + 2]);
		}
	}

	// compute the covariance Matrix
	cvZero(covarienceOfMat);
	cvZero(avgOfMat);
	cvCalcCovarMatrix((const void**)&mat, 1, covarienceOfMat, avgOfMat, CV_COVAR_NORMAL | CV_COVAR_ROWS);
	if (M > 1) cvConvertScale(covarienceOfMat, covarienceOfMat, 1.0 / (M - 1)); // normalization
	cvReleaseMat(&mat);*/
}

void   Imagematting::getCiCj(float* mat, int i, int j) // store RGB of (i, j)-th pixel in mat
{
	/*cvmSet(mat, 0, 0, data[i * step + j * channels]);
	cvmSet(mat, 0, 1, data[i * step + j * channels + 1]);
	cvmSet(mat, 0, 2, data[i * step + j * channels + 2]);*/
}

void   Imagematting::getWeight2() // get local smooth term Wlap(ij)
{
	// get Ci, Cj, uk, sigmaK of every 3*3 window
	/*std::vector<T> triplets;
	double w;
	CvMat* reverseMat = cvCreateMat(3, 3, CV_64FC1); // save the matrix which needs to be reverse
	CvMat* IdenMat = cvCreateMat(3, 3, CV_64FC1); // the identity mat
	CvMat* CiMat = cvCreateMat(1, 3, CV_64FC1); // for Ci
	CvMat* CjMat = cvCreateMat(1, 3, CV_64FC1); // for Cj
	CvMat* stoMat = cvCreateMat(1, 3, CV_64FC1); // store mat
	CvMat* stoMat2 = cvCreateMat(1, 3, CV_64FC1); // store mat
	CvMat* stoMattr = cvCreateMat(3, 1, CV_64FC1); // store mat
	CvMat* result = cvCreateMat(1, 1, CV_64FC1); // the result mat
	cvSetIdentity(IdenMat); //get I
	double wi = winSize * winSize;
	for (int i = 0; i < height - winSize + 1; i++) //(i, j) is the up left point of every window
	{
		for (int j = 0; j < width - winSize + 1; j++)
		{
			// compute uk and sigmaK(update covarienceOfMat and avgOfMat)
			getCovarianceMatrix(i, j); //send the up-left point
			cvAddWeighted(covarienceOfMat, 1, IdenMat, REG / wi, 0, reverseMat); //covarienceOfMat + REG/9 * I
			cvInvert(reverseMat, reverseMat, CV_SVD_SYM); //Mat = (Mat)-1

			uchar* ddata = (uchar*)dilateimg->imageData;
			//Wij = Wji
			//get W2 in horizontal direction
			for (int l1 = 0; l1 < winSize; l1++)
			{
				for (int l2 = 0; l2 < winSize - 1; l2++)
				{
					// (l1 + i, l2 + j) & (l1 + i, l2 + 1 + j)
					int x1 = l1 + i;
					int y1 = l2 + j;
					int x2 = l1 + i;
					int y2 = l2 + 1 + j;
					int binary1 = ddata[x1 * g_step + y1];
					int binary2 = ddata[x2 * g_step + y2];
					if (binary1 != 0 || binary2 != 0) // binary1 and binary 2
					{
						getCiCj(CiMat, x1, y1); // store RGB of (x1, y1)-th pixel in CiMat
						getCiCj(CjMat, x2, y2);
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1
						cvSub(CjMat, avgOfMat, stoMat2); // Cj - uk
						cvTranspose(stoMat2, stoMattr); // T(Cj - uk)const
						cvMatMul(stoMat, stoMattr, result); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1 * T(Cj - uk)
						w = 1 + cvmGet(result, 0, 0);
						triplets.push_back(T(x1 * width + y1 + 2, x2 * width + y2 + 2, -delta / wi * w)); // Wij
						triplets.push_back(T(x1 * width + y1 + 2, x1 * width + y1 + 2, delta / wi * w)); // add to L(i, i)
						triplets.push_back(T(x2 * width + y2 + 2, x1 * width + y1 + 2, -delta / wi * w)); // Wji
						triplets.push_back(T(x2 * width + y2 + 2, x2 * width + y2 + 2, delta / wi * w)); // add to L(i, i)
					}
				}
			}

			//get W2 in vertical direction
			for (int l1 = 0; l1 < winSize - 1; l1++)
			{
				for (int l2 = 0; l2 < winSize; l2++)
				{
					// (l1 + i, l2 + j) & (l1 + 1 + i, l2 + j)
					int x1 = l1 + i;
					int y1 = l2 + j;
					int x2 = l1 + 1 + i;
					int y2 = l2 + j;
					int binary1 = ddata[x1 * g_step + y1];
					int binary2 = ddata[x2 * g_step + y2];
					if (binary1 != 0 && binary2 != 0)
					{
						getCiCj(CiMat, x1, y1); // store RGB of (x1, y1)-th pixel in CiMat
						getCiCj(CjMat, x2, y2);
						cvSub(CiMat, avgOfMat, stoMat);//Ci - uk
						cvMatMul(stoMat, reverseMat, stoMat); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1
						cvSub(CjMat, avgOfMat, stoMat2); // Cj - uk
						cvTranspose(stoMat2, stoMattr); // T(Cj - uk)const
						cvMatMul(stoMat, stoMattr, result); // (Ci - uk) * (covarienceOfMat + REG/9 * I)-1 * T(Cj - uk)
						w = 1 + cvmGet(result, 0, 0);
						triplets.push_back(T(x1 * width + y1 + 2, x2 * width + y2 + 2, -delta / wi * w)); // Wij
						triplets.push_back(T(x1 * width + y1 + 2, x1 * width + y1 + 2, delta / wi * w)); // add to L(i, i)
						triplets.push_back(T(x2 * width + y2 + 2, x1 * width + y1 + 2, -delta / wi * w)); // Wji
						triplets.push_back(T(x2 * width + y2 + 2, x2 * width + y2 + 2, delta / wi * w)); // add to L(i, i)
					}
				}
			}
		}
	}
	cvReleaseMat(&reverseMat);
	cvReleaseMat(&IdenMat);
	cvReleaseMat(&CiMat);
	cvReleaseMat(&CjMat);
	cvReleaseMat(&stoMat);
	cvReleaseMat(&stoMat2);
	cvReleaseMat(&stoMattr);
	cvReleaseMat(&result);
	W2.setFromTriplets(triplets.begin(), triplets.end());
	W2.prune(0.0);*/
	printf("getWeight2 ok\n");
}

void   Imagematting::getWeight3() // get unlocal smooth term Wlle(ij), use LLE
{
	/*std::vector<T> triplets;
	double del = 0.01 * 0.01 / K; // add a very small number to eigenvalue of XtX in order to insurance invertibility of XtX

	for (int i = 0; i < allsize; i++)
	{
		int x = AT_(allmat, i, 0);
		int y = AT_(allmat, i, 1);
		VectorXd Y(5, 1);
		MatrixXd X(K + 1, 5);
		MatrixXd XtX(K, K); // Xt is the transposition of X
		MatrixXd I(K, K); // identity matrix
		VectorXd W(K, 1); // need to compute
		I = MatrixXd::Identity(K, K);
		for (int j = 0; j < 5; j++) Y(j) = AT(allmat, i, j);
		for (int j = 0; j < K + 1; j++) // search the K-nearest pixels in RGBXY
		{
			int index = AT(allresult.indices, i, j); // index + 2 is the index in N
			X.row(j) << AT(allmat, index, 0), AT(allmat, index, 1), AT(allmat, index, 2), AT(allmat, index, 3), AT(allmat, index, 4);
			X.row(j) = Y.transpose() - X.row(j);
		}

		// remove itself from K+1 neighbors, only remove one row
		int flag = K; // flag is the index of the removed neighbor
		VectorXd ze = VectorXd::Zero(5);
		for (int j = 0; j < K + 1; j++)
		{
			if (X.row(j) == ze.transpose()) // all zeros
			{
				// remove this row
				if (j < K) X.block(j, 0, K - j, 5) = X.block(j + 1, 0, K - j, 5);
				X.conservativeResize(K, 5);
				flag = j;
				break;
			}
			if (j == K) X.conservativeResize(K, 5); // if no zero row, remove the last row
		}

		// now only K neighbors
		XtX = X * X.transpose();
		double tr = XtX.trace();
		XtX = XtX + del * I * tr;
		XtX = XtX.inverse();

		for (int j = 0; j < K; j++) // search the K-nearest pixels in RGBXY
		{
			W(j) = XtX.row(j).sum() / XtX.sum(); // normalize
			int  Knearest;  // Knearest is the index of K-nearest neighbors of i
			if (flag > j) // ignore the removed neighbor
				Knearest = AT(allresult.indices, i, j);
			else
				Knearest = AT(allresult.indices, i, j + 1);
			triplets.push_back(T(x * width + y + 2, getBigIndex(Knearest), -W(j))); // add to L(i, j)
		}
		triplets.push_back(T(x * width + y + 2, x * width + y + 2, 1)); // add to L(i, i)
	}
	W3.setFromTriplets(triplets.begin(), triplets.end());
	W3.prune(0.0);*/
	printf("getWeight3 ok\n");
}

int    Imagematting::getBigIndex(int i) // i is index in allmat, return index in Alpha
{
	int x = AT_(allmat, i, 0);
	int y = AT_(allmat, i, 1);
	return x * width + y + 2; // the first and second element of Alpha are two virtue nodes
}

void   Imagematting::getG()
{
	// Gi is set to 1 if i belongs to foreground and unknown pixel which confidence > CONFI, 0 otherwise
	int highconfi = 0; // number of confidence > CONFI
	for (int i = 0; i < allsize; i++) // 0, 1 are two virtue nodes
	{
		// int x = i / width;
		//int y = i - x * width;
		int x = AT_(allmat, i, 0);
		int y = AT_(allmat, i, 1);
		if (tri[x][y] == 1) // foreground
			G[x * width + y + 2] = 1;
		else if (confidence[x][y] > CONFI)
		{
			G[x * width + y + 2] = preAlpha[x][y];
			highconfi++;
		}
	}
	printf("confi > CONFI:  = %d    unknown size: %d\n", highconfi, usize);
	printf("getG ok\n");
}

void   Imagematting::getI()
{
	/*// Iii = lambda_E if i belongs to S(S = f + b + u(confidence > CONFI))
	std::vector<T> triplets;
	//triplets.push_back(T(0, 0, lambda_E));  // let two virtue nodes be lambda_E (I00 = I11 = lambda_E)
	//triplets.push_back(T(1, 1, lambda_E));
	for (int i = 0; i < allsize; i++)
	{
		//int x = i / width;
		//int y = i - x * width;
		int x = AT_(allmat, i, 0);
		int y = AT_(allmat, i, 1);
		if (tri[x][y] == 0 || tri[x][y] == 1 || confidence[x][y] > CONFI)
			triplets.push_back(T(x * width + y + 2, x * width + y + 2, lambda_E));
	}
	I.setFromTriplets(triplets.begin(), triplets.end());*/
	printf("getI ok\n");
}

void   Imagematting::getL() // get unlocal smooth term Wlle(ij)
{
	/*// L = W1 + W2 + W3
	getWeight1();
	getWeight2();
	getWeight3();
	L = W1 + W2 + W3;
	L.prune(0.0);*/
	printf("getL ok\n");
}

void   Imagematting::getFinalAlpha()
{
	getI();
	getG();
	getL();
	/*// (I + T(L) * L) * alpha = I * G
	SpMat A = I + (L.transpose() * L);
	A.prune(0.0); // delete zero elements

	// save A
	// fstream f("A.txt", ios::out);
	// if (!f) cout << "Error!" << endl;
	// for (int k = 0; k < A.outerSize(); ++k)
	//   {
	//       SpMat::InnerIterator it(A, k);
	//       for (; it; ++it)
	//       {
	//         f << it.value() << "   ";
	//       }
	//   }
	// f.close();

	// compute the sparsity of matrix A
	cout << "The size of A: (" << A.rows() << ", " << A.cols() << ")\n";
	cout << "The non-zero numbers of matrix A: " << A.nonZeros() << endl;
	cout << "The sparsity of A:" << A.nonZeros() / double(A.rows()) / double(A.cols()) << endl;

	VectorXd b = I * G;

	clock_t start, finish;
	start = clock();

	ConjugateGradient<SpMat> cg(A); // use CG method
	Alpha.setZero();
	Alpha = cg.solve(b);

	// SimplicialLDLT<SpMat> ldlt(A); // use LDLT method
	// Alpha = ldlt.solve(b); // all alpha are more than 0

	finish = clock();
	cout << double(finish - start) / CLOCKS_PER_SEC << endl;*/

	printf("getFinalAlpha ok\n");
}

void Imagematting::showMatte()
{
	//  cout << Alpha[0] << "," << Alpha[1] << endl; // show computed alpha of two virtue nodes
	unsigned char* udata = (unsigned char*)matte;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			if (tri[i][j] == 0) // background
				udata[i * g_step + j] = 0;
			else if (tri[i][j] == 1) // foreground
				udata[i * g_step + j] = 255;
			else // unknown area
			{
				int index = i * width + j + 2;
				udata[i * g_step + j] = MAX(MIN(int(Alpha[index] * 255), 255), 0); // notice: abs is not right
				//udata[i * g_step + j] = MAX(MIN(int(confidence[i][j] * 255 * 5), 255), 0); // use preAlpha
			}
		}
	}
}

void Imagematting::solveAlpha()
{
	findKnearest(); //get K nearest backgrounds(indices + dists)

	//// read four mats in "Kdatas.xml"
	//FileStorage fs("K1.xml", FileStorage::READ);
	//fs["findices"] >> fresult.indices;
	//fs["bindices"] >> bresult.indices;
	//fs["allindices"] >> allresult.indices;
	//fs.release();

	getPreAlpha(); // get predicted alpha of every pixel

	//// get array preAlpha
	//fstream f("a1.txt", ios::in);
	//for (int i = 0; i < height; i++)
	//{
	//for (int j = 0; j < width; j++)
	//{
	//f >> preAlpha[i][j];
	//}
	//}

	getFinalAlpha();
	showMatte();
}