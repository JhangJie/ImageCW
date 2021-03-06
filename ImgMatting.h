#include "Mat.h"

#define K          10 // the nearst K neighbors in RGBXY space
#define winSize    3 // the size of a neighbor matrix
#define sigma      0.1 // variance
#define delta      0.1 // the coe of W2
#define gamma      0.1 // the coe of W1
#define REG        1e-4
#define CONFI      0.18
#define lambda_E   1000
#define di_time    5 // dilation times

//#define AT(mat, x, y) mat.at<int>(x, y)
#include <vector>

//typedef Eigen::SparseMatrix<double> SpMat;
//typedef Eigen::Triplet<double> T;

struct kdResult // the result of knnsearch
{
    float* indices;
    float* dists;
};

class Imagematting
{
private:
    unsigned char* img;		    // the original image
    unsigned char* trimap;	    // the original trimap
    unsigned char* matte;	    // the result image
    unsigned char* dilateimg;   // after dilation

    int      height;
    int      width;
    int      step;   // widthstep of image
    int      g_step; // widthstep of gray
    int      channels;

    unsigned char* data;
    int      N;      // N = height * width + 2; order: VirtueF, VirtueB, all nodes ((0, 0), !-!-, (height - 1, width - 1))

    int      bsize;     // size of background pixels
    int      fsize;     // size of foreground pixels
    int      usize;     // size of unknown pixels
    int      allsize;   // siez of dilating pixels

    float* bmat;	// mat of background pixels
    float* fmat;	// mat of foreground pixels
    float* umat;    // mat of unknown pixels
    float* allmat;	// mat of all pixels

    int* dB;    // the min color distance between C and B
    int* dF;    // the min color distance between C and F
    int** tri;  // 1 is foreground, 0 is background, 2 is unknown
    //  int      **xy_index; // (x,y) get index in allmat
    double** preAlpha;       // mat of predicted alpha (n * 1 matrix)
    double** confidence;     // confidence for every pixel

    float* covarienceOfMat; // covarience Matrix in 3*3 window
    float* avgOfMat;        // average Matrix in 3*3 window

    kdResult bresult, fresult, allresult; // the results of kd-tree by using FLANN

    // for eigen
    matrix_t* W1;    // save W(i, F) & W(i, B) in a big SparseMatrix W1(N * N)
    matrix_t* W2;    // save local smooth term Wlap(ij) in a big SparseMatrix W2(N * N)
    matrix_t* W3;    // save unlocal smooth term Wlle(ij) in a big SparseMatrix W3(N * N)
    matrix_t* L;     // L = -W1 - W2 - W3
    matrix_t* I;     // I is a diagonal Matrix

    unsigned char* G;     // G is a Vector
    unsigned char* Alpha; // the final alpha

    void     dilate(unsigned char* inptr, int w, int h);
    void     addInMat(float* mat, int n, int i, int j, int b, int g, int r); // add a RGBXY parameter in allmat
    void     addInMat(float* mat, int n, int x, int y); // add a XY parameter in  mat
    void     createMat(); // create b-, f-, u-, allmat

    void     getD();                                //the minimum distances between foreground/background sample and the current pixel
    double   geteveryAlpha(int c, int f, int b);    // get predicted alpha of C, use it fth-nearest foreground pixel and bth-nearest background pixel
    double   getRd(int c, int f, int b);            // get Rd of C, use it fth-nearest foreground pixel and bth-nearest background pixel
    double   getW(int c, int fb, bool flag);        // get W of C, fb is the fbth-nearest fore- or background pixel of C
    double   getConfidence(int c, int f, int b);    // c is the index in umat, f is the index of C's f-nearest foreground pixel,  b is the index of C's b-nearest background pixel

    void     getWeight1(); // get data term W(i, F) & W(i, B)
    void     getWeight2(); // get local smooth term Wlap(ij)
    void     getWeight3(); // get unlocal smooth term Wlle(ij)

    void     getCovarianceMatrix(int i, int j);         // get covariance matrix of a 3*3 window, x & y are the middle points in one 3*3 window
    void     getCiCj(float* mat, int i, int j); // get the (i,j)-th RGB of data in mat

    int      getBigIndex(int i); // i is index in allmat, return index in Alpha

    void     getG(); // get G
    void     getI(); // get I
    void     getL(); // get L

    int      BC(float* mat, int index);
    int      GC(float* mat, int index);
    int      RC(float* mat, int index);

    double   getEc(int c, int f, int b);
    double   getEs(int c, int fb, bool flag);
    void     getDspace();

    int* dsB; // the min spacial distance between C and B
    int* dsF; // the min spacial distance between C and F

public:
    Imagematting();
    ~Imagematting();

    //void     loadImage(char * filename);
    //void     loadTrimap(char * filename);
    void     loadImage(unsigned char* inptr, int w, int h);
    void     loadTrimap(unsigned char* inptr, int w, int h);

    void     findKnearest();    // find K nearest pixels of every pixel
    void     getPreAlpha();     // get predicted Alpha Matrix named preAlpha
    void     getFinalAlpha();   // use QR to solve Ax = b

    void     save(char* filename); // save the matting result
    void     solveAlpha();
    void     showMatte();

    //just for test
    //void     TEST(SpMat A);
};
