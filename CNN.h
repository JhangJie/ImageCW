#include "CNNMat.h"
#include "minst.h"

#define AvePool 0
#define MaxPool 1
#define MinPool 2

//卷積層
typedef struct convolutional_layer 
{
	int inputWidth;
	int inputHeight;
	int mapSize;      //特徵模板的大小，模板一般都是正方形

	int inChannels;   //輸入圖像的數量
	int outChannels;  //輸出圖像的數量

	//關於特徵模板的權重分佈，這裡是一個四維數組
	//其大小為inChannels * outChannels * mapSize * mapSize大小
	//這裡用四維整數，主要是為了表現全連接的形式，實際上卷積層並沒有用到全連接的形式
	//這裡的例子是DeapLearningToolboox裡的CNN例子，其用到就是全連接
	float**** mapData;     //存放特徵模塊的數據
	float**** dmapData;    //存放特徵模塊的數據的局部梯度

	float* basicData;	   //預期，調整的大小，為outChannels
	bool isFullConnect;    //是否為全連接
	bool* connectModel;    //連接模式（或為全連接）

	//下面三者的大小同輸出的尺寸相同
	float*** v; //進入激活函數的輸入值
	float*** y; //激活函數後神經元的輸出

	//輸出結果的局部梯度
	float*** d; //網絡的局部梯度，δ值  
}CovLayer;

//采样层 pooling
typedef struct pooling_layer 
{
	int inputWidth;   //输入图像的宽
	int inputHeight;  //输入图像的长
	int mapSize;      //特征模板的大小

	int inChannels;   //输入图像的数目
	int outChannels;  //输出图像的数目

	int poolType;     //Pooling的方法
	float* basicData; //偏置

	float*** y; //采样函数后神经元的输出,无激活函数
	float*** d; //网络的局部梯度,δ值
}PoolLayer;

//输出层 全连接的神经网络
typedef struct nn_layer 
{
	int inputNum;		//输入数据的数目
	int outputNum;		//输出数据的数目

	float** wData;		//权重数据，为一个inputNum*outputNum大小
	float* basicData;   //偏置，大小为outputNum大小

	// 下面三者的大小同输出的维度相同
	float* v;	//进入激活函数的输入值
	float* y;	//激活函数后神经元的输出
	float* d;	//网络的局部梯度,δ值

	bool isFullConnect; //是否为全连接
}OutLayer;

typedef struct cnn_network 
{
	int layerNum;
	CovLayer* C1;
	PoolLayer* S2;
	CovLayer* C3;
	PoolLayer* S4;
	OutLayer* O5;

	float* e; //训练误差
	float* L; //瞬时误差能量
}CNN;

typedef struct train_opts 
{
	int numepochs;	//训练的迭代次数
	float alpha;	//学习速率
}CNNOpts;

class CNNNet
{
private:
	int vecmaxIndex(float* vec, int veclength);			//返回向量最大数的序号

	float vecMulti(float* vec1, float* vec2, int vecL);	//两向量相乘
	float sigma_derivation(float y);					//Logic激活函数的自变量微分

	//初始化卷积层
	CovLayer* initCovLayer(int inputWidth, int inputHeight, int mapSize, int inChannels, int outChannels);
	void CovLayerConnect(CovLayer* covL, bool* connectModel);

	//初始化采样层
	PoolLayer* initPoolLayer(int inputWidth, int inputHeigh, int mapSize, int inChannels, int outChannels, int poolType);
	void PoolLayerConnect(PoolLayer* poolL, bool* connectModel);

	//初始化输出层
	OutLayer* initOutLayer(int inputNum, int outputNum);

	//激活函数 input是数据，inputNum说明数据数目，bas表明偏置
	float activation_Sigma(float input, float bas); //sigma激活函数

	void cnnff(CNN* cnn, float** inputData);		//网络的前向传播
	void cnnbp(CNN* cnn, float* outputData);		//网络的后向传播
	void cnnapplygrads(CNN* cnn, CNNOpts opts, float** inputData);
	void cnnclear(CNN* cnn);						//将数据vyd清零

	/*
		Pooling Function
		input 输入数据
		inputNum 输入数据数目
		mapSize 求平均的模块区域
	*/
	void avgPooling(float** output, nSize outputSize, float** input, nSize inputSize, int mapSize); //求平均值

	/*
		单层全连接神经网络的处理
		nnSize是网络的大小
	*/
	void nnff(float* output, float* input, float** wdata, float* bas, nSize nnSize); //单层全连接神经网络的前向传播
	void savecnndata(CNN* cnn, const char* filename, float** inputdata);			 //保存CNN网络中的相关数据

public:
	CNNNet();
	~CNNNet();

	//Test
	void test_minst();
	void test_mat1();
	void test_cnn();

	void cnnsetup(CNN* cnn, nSize inputSize, int outputSize);
	/*
		CNN网络的训练函数
		inputData，outputData分别存入训练数据
		dataNum表明数据数目
	*/
	void cnntrain(CNN* cnn, ImgArr inputData, LabelArr outputData, CNNOpts opts, int trainNum);
	float cnntest(CNN* cnn, ImgArr inputData, LabelArr outputData, int testNum);	//测试cnn函数

	void savecnn(CNN* cnn, const char* filename);		//保存cnn
	void importcnn(CNN* cnn, const char* filename);		//导入cnn的数据

};