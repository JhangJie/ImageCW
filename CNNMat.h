#define full 0
#define same 1
#define valid 2

typedef struct Mat2DSize
{
    int c; // �C w
    int r; // �� h
}nSize;

class CNNMat
{
private:


public:
    CNNMat();
    ~CNNMat();

    float** rotate180(float** mat, nSize matSize);  //�x�}����180��
    void addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2);       //�x�}�ۥ[

    float** correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type); //������
    float** cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type);         //���n�ާ@

    float** UpSample(float** mat, nSize matSize, int upc, int upr);     //�o�ӬO�x�}���W�ļˡ]���Ȥ����^�Aupc��upr�O��������

    float** matEdgeExpand(float** mat, nSize matSize, int addc, int addr);          //���G���x�}��t�X�j�A�W�[addw�j�p��0����
    float** matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr);    //���G���x�}��t�Y�p�A�վ�shrinkc�j�p����

    void savemat(float** mat, nSize matSize, const char* filename);             //�O�s�x�}�ƾ�
    void multifactor(float** res, float** mat, nSize matSize, float factor);    //�x�}���H�Y��
    float summat(float** mat, nSize matSize);                                   //�x�}�U�������M

};
