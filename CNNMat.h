#define full 0
#define same 1
#define valid 2

typedef struct Mat2DSize
{
    int c; // 列 w
    int r; // 行 h
}nSize;

class CNNMat
{
private:


public:
    CNNMat();
    ~CNNMat();

    float** rotate180(float** mat, nSize matSize);  //矩陣旋轉180度
    void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);       //矩陣相加

    float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type); //互相關
    float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type);         //卷積操作

    float** UpSample(float** mat, nSize matSize, int upc, int upr);     //這個是矩陣的上採樣（等值內插），upc及upr是內插倍數

    float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);          //給二維矩陣邊緣擴大，增加addw大小的0值邊
    float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr);    //給二維矩陣邊緣縮小，調整shrinkc大小的邊

    void savemat(float** mat, nSize matSize, const char* filename);             //保存矩陣數據
    void multifactor(float** res, float** mat, nSize matSize, float factor);    //矩陣乘以係數
    float summat(float** mat, nSize matSize);                                   //矩陣各元素的和

};
