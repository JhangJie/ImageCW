#include "pch.h"
#include "CNNMat.h"
#include <stdlib.h>
#include <malloc.h>
#include "math.h"
#include <direct.h>
#include <ctype.h>
#include <io.h>
#include <iostream> 
#include <fstream>
#include <string> 
#include <sstream> 
#include <assert.h>

CNNMat::CNNMat()
{

}

CNNMat::~CNNMat()
{

}

float** CNNMat::rotate180(float** mat, nSize matSize)   //矩陣旋轉180度
{
    int i, c, r;
    int outSizeW = matSize.c;
    int outSizeH = matSize.r;
    float** outputData = (float**)malloc(outSizeH * sizeof(float*));
    for (i = 0; i < outSizeH; i++)
        outputData[i] = (float*)malloc(outSizeW * sizeof(float));

    for (r = 0; r < outSizeH; r++)
        for (c = 0; c < outSizeW; c++)
            outputData[r][c] = mat[outSizeH - r - 1][outSizeW - c - 1];

    return outputData;
}

//關於卷積和相關操作的輸出選項
//這里共有三種選擇：完全，相同，有效，分別表示
//full指完全，操作後結果的大小為inSize +（mapSize-1）
//same指同輸入相同大小
//有效指完全操作後的大小，一般為inSize-（mapSize-1）大小，其不需要將輸入添0擴大。

float** CNNMat::correlation(float** map, nSize mapSize, float** inputData, nSize inSize, int type)  //互相關
{
    //這裡的互相關是在後向傳播時調用，或者將地圖反轉180度再卷積
    //為了方便計算，這裡先將圖像擴展一圈
    //這裡的捲積要分為兩撥，偶數模板同奇數模板
    int i, j, c, r;
    int halfmapsizew;
    int halfmapsizeh;
    if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)   //模板大小為偶數
    {
        halfmapsizew = (mapSize.c) / 2;             //卷積模塊的半瓣大小
        halfmapsizeh = (mapSize.r) / 2;
    }
    else
    {
        halfmapsizew = (mapSize.c - 1) / 2;         //卷積模塊的半瓣大小
        halfmapsizeh = (mapSize.r - 1) / 2;
    }

    //這裡先進行進行全模式的操作，全模式的輸出大小為inSize +（mapSize-1）
    int outSizeW = inSize.c + (mapSize.c - 1);      //這裡的輸出擴大一部分
    int outSizeH = inSize.r + (mapSize.r - 1);
    float** outputData = (float**)malloc(outSizeH * sizeof(float*)); //互相關的結果擴大了
    for (i = 0; i < outSizeH; i++)
        outputData[i] = (float*)calloc(outSizeW, sizeof(float));

    //為了方便計算，將inputData擴大一圈
    float** exInputData = matEdgeExpand(inputData, inSize, mapSize.c - 1, mapSize.r - 1);

    for (j = 0; j < outSizeH; j++)
        for (i = 0; i < outSizeW; i++)
            for (r = 0; r < mapSize.r; r++)
                for (c = 0; c < mapSize.c; c++)
                    outputData[j][i] = outputData[j][i] + map[r][c] * exInputData[j + r][i + c];

    for (i = 0; i < inSize.r + 2 * (mapSize.r - 1); i++)
        free(exInputData[i]);
    free(exInputData);

    nSize outSize = { outSizeW,outSizeH };
    switch (type)   //根據不同的情況，返回不同的結果
    {
    case full:      //完全大小的情况
        return outputData;
    case same:
    {
        float** sameres = matEdgeShrink(outputData, outSize, halfmapsizew, halfmapsizeh);
        for (i = 0; i < outSize.r; i++)
            free(outputData[i]);
        free(outputData);
        return sameres;
    }
    case valid:
    {
        float** validres;
        if (mapSize.r % 2 == 0 && mapSize.c % 2 == 0)
            validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2 - 1, halfmapsizeh * 2 - 1);
        else
            validres = matEdgeShrink(outputData, outSize, halfmapsizew * 2, halfmapsizeh * 2);
        for (i = 0; i < outSize.r; i++)
            free(outputData[i]);
        free(outputData);
        return validres;
    }
    default:
        return outputData;
    }
}

float** CNNMat::cov(float** map, nSize mapSize, float** inputData, nSize inSize, int type)  //卷積操作
{
    //卷積操作可以用旋轉180度的特徵模板相關來求
    float** flipmap = rotate180(map, mapSize); //旋轉180度的特徵模板
    float** res = correlation(flipmap, mapSize, inputData, inSize, type);
    int i;
    for (i = 0; i < mapSize.r; i++)
        free(flipmap[i]);
    free(flipmap);
    return res;
}

//這個是矩陣的上採樣（等值內插），upc及upr是內插倍數
float** CNNMat::UpSample(float** mat, nSize matSize, int upc, int upr)
{
    int i, j, m, n;
    int c = matSize.c;
    int r = matSize.r;
    float** res = (float**)malloc((r * upr) * sizeof(float*)); //結果的初始化
    for (i = 0; i < (r * upr); i++)
        res[i] = (float*)malloc((c * upc) * sizeof(float));

    for (j = 0; j < r * upr; j = j + upr)
    {
        for (i = 0; i < c * upc; i = i + upc)   //寬的擴充
            for (m = 0; m < upc; m++)
                res[j][i + m] = mat[j / upr][i / upc];

        for (n = 1; n < upr; n++)               //高的擴充
            for (i = 0; i < c * upc; i++)
                res[j + n][i] = res[j][i];
    }
    return res;
}

//給二維矩陣邊緣擴大，增加addw大小的0值邊
float** CNNMat::matEdgeExpand(float** mat, nSize matSize, int addc, int addr)
{   //向量邊緣擴大
    int i, j;
    int c = matSize.c;
    int r = matSize.r;
    float** res = (float**)malloc((r + 2 * addr) * sizeof(float*)); //結果的初始化

    for (i = 0; i < (r + 2 * addr); i++)
        res[i] = (float*)malloc((c + 2 * addc) * sizeof(float));

    for (j = 0; j < r + 2 * addr; j++)
    {
        for (i = 0; i < c + 2 * addc; i++)
        {
            if (j < addr || i < addc || j >= (r + addr) || i >= (c + addc))
                res[j][i] = (float)0.0;
            else
                res[j][i] = mat[j - addr][i - addc];                //複製原向量的數據
        }
    }
    return res;
}

//給二維矩陣邊緣縮小，預定shrinkc大小的邊
float** CNNMat::matEdgeShrink(float** mat, nSize matSize, int shrinkc, int shrinkr)
{   //實際上的縮小，寬縮小addw，高縮小addh
    int i, j;
    int c = matSize.c;
    int r = matSize.r;
    float** res = (float**)malloc((r - 2 * shrinkr) * sizeof(float*)); //結果矩陣的初始化

    for (i = 0; i < (r - 2 * shrinkr); i++)
        res[i] = (float*)malloc((c - 2 * shrinkc) * sizeof(float));

    for (j = 0; j < r; j++)
    {
        for (i = 0; i < c; i++)
        {
            if (j >= shrinkr && i >= shrinkc && j < (r - shrinkr) && i < (c - shrinkc))
                res[j - shrinkr][i - shrinkc] = mat[j][i];             //複製原向量的數據
        }
    }
    return res;
}

void CNNMat::savemat(float** mat, nSize matSize, const char* filename)
{
	FILE* fp = NULL;
    fp = fopen(filename, "wb");
    if (fp == NULL)
        printf("write file failed\n");

    int i;
    for (i = 0; i < matSize.r; i++)
        fwrite(mat[i], sizeof(float), matSize.c, fp);
    fclose(fp);
}

void CNNMat::addmat(float** res, float** mat1, nSize matSize1, float** mat2, nSize matSize2)    //矩陣相加
{
    int i, j;
    if (matSize1.c != matSize2.c || matSize1.r != matSize2.r)
        printf("ERROR: Size is not same!");

    for (i = 0; i < matSize1.r; i++)
        for (j = 0; j < matSize1.c; j++)
            res[i][j] = mat1[i][j] + mat2[i][j];
}

void CNNMat::multifactor(float** res, float** mat, nSize matSize, float factor) //矩陣乘以係數
{
    int i, j;
    for (i = 0; i < matSize.r; i++)
        for (j = 0; j < matSize.c; j++)
            res[i][j] = mat[i][j] * factor;
}

float CNNMat::summat(float** mat, nSize matSize)    //矩陣各元素的和
{
    float sum = 0.0;
    int i, j;
    for (i = 0; i < matSize.r; i++)
        for (j = 0; j < matSize.c; j++)
            sum = sum + mat[i][j];
    return sum;
}