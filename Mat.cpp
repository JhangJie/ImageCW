#include "pch.h"
#include "ImgProcessing.h"
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
#include <assert.h>
//https://michaelchen.tech/data-structures-in-c/sparse-matrix-in-array/

CMat::CMat()
{

}

CMat::~CMat()
{

}

void CMat::matrix_delete(void* self)
{
    assert(self);
    size_t* cols = ((matrix_t*)self)->cols;
    if (cols) free(cols);

    size_t* rows = ((matrix_t*)self)->rows;
    if (rows) free(rows);

    double* elements = ((matrix_t*)self)->elements;
    if (elements) free(elements);
    free(self);
}

matrix_t* CMat::matrix_new(size_t col, size_t row)
{
    assert(0 < col);
    assert(0 < row);

    matrix_t* mtx = new matrix_t[sizeof(matrix_t)];
    //matrix_t* mtx = malloc(sizeof(matrix_t));
    if (!mtx) return mtx;

    mtx->col = col;
    mtx->row = row;
    mtx->size = 0;
    mtx->capacity = 2;

    mtx->cols = (size_t*)calloc(mtx->capacity, sizeof(size_t));
    if (!(mtx->cols)) 
    {
        matrix_delete(mtx);
        mtx = NULL;
        return mtx;
    }

    mtx->rows = (size_t*)calloc(mtx->capacity, sizeof(size_t));
    if (!(mtx->rows)) 
    {
        matrix_delete(mtx);
        mtx = NULL;
        return mtx;
    }

    mtx->elements = (double*)calloc(mtx->capacity, sizeof(double));
    if (!(mtx->elements)) 
    {
        matrix_delete(mtx);
        mtx = NULL;
        return mtx;
    }
    return mtx;
}

size_t CMat::matrix_col(const matrix_t* self) //取得稀疏矩陣的大小 (或維度)
{
    assert(self);
    return self->col;
}

size_t CMat::matrix_row(const matrix_t* self)
{
    assert(self);
    return self->row;
}

double CMat::matrix_at(const matrix_t* self, size_t col, size_t row)  //取得稀疏矩陣中任意位置的值
{
    assert(col < matrix_col(self));
    assert(row < matrix_row(self));

    for (size_t i = 0; i < self->size; i++)
        if (self->cols[i] == col && self->rows[i] == row)
            return self->elements[i];
    return 0.0;
}

bool CMat::matrix_is_equal(const matrix_t* m, const matrix_t* n)  //確認兩矩陣相等
{
    if (matrix_col(m) != matrix_col(n)) return false;
    if (matrix_row(m) != matrix_row(n)) return false;

    for (size_t i = 0; i < matrix_col(m); i++)
        for (size_t j = 0; j < matrix_row(m); j++)
            if (fabs(matrix_at(m, i, j) - matrix_at(n, i, j)) > 0.000001)
                return false;
    return true;
}

void CMat::matrix_delete_at(matrix_t* self, size_t col, size_t row)
{
    assert(col < matrix_col(self));
    assert(row < matrix_row(self));

    if (self->size <= 1) 
    {
        self->size--;
        return;
    }

    size_t i = 0;
    bool matched = false;
    while (i < self->size - 1) 
    {
        if (self->cols[i] == col && self->rows[i] == row)
            matched = true;

        if (matched) 
        {
            self->cols[i] = self->cols[i + 1];
            self->rows[i] = self->rows[i + 1];
            self->elements[i] = self->elements[i + 1];
        }
        i++;
    }
    self->size--;
}

bool CMat::matrix_expand(matrix_t* self)
{
    assert(self);
    if (self->size < self->capacity) return true;

    self->capacity <<= 1;

    size_t* cols = (size_t*)malloc(self->capacity * sizeof(size_t));
    if (!cols) return false;

    size_t* rows = (size_t*)malloc(self->capacity * sizeof(size_t));
    if (!rows) 
    {
        free(cols);
        return false;
    }

    double* elements = (double*)malloc(self->capacity * sizeof(double));
    if (!elements) 
    {
        free(cols);
        free(rows);
        return false;
    }

    size_t i = 0;
    while (i < self->size) 
    {
        cols[i] = self->cols[i];
        rows[i] = self->rows[i];
        elements[i] = self->elements[i];
        i++;
    }

    free(self->cols);
    self->cols = cols;
    free(self->rows);
    self->rows = rows;
    free(self->elements);
    self->elements = elements;
    return true;
}

bool CMat::matrix_set_at(matrix_t* self, size_t col, size_t row, double value)    //存入稀疏矩陣中任意位置的值
{
    assert(col < matrix_col(self));
    assert(row < matrix_row(self));

    for (size_t i = 0; i < self->size; i++) 
    {
        if (self->cols[i] == col && self->rows[i] == row) 
        {
            if (fabs(value) > 0.000001) 
                self->elements[i] = value;
            else 
                matrix_delete_at(self, col, row);
            return true;
        }
    }

    if (fabs(value) < 0.000001) return true;
    if (!matrix_expand(self)) return false;

    self->cols[self->size] = col;
    self->rows[self->size] = row;
    self->elements[self->size] = value;
    self->size++;
    return true;
}

matrix_t* CMat::matrix_trans(const matrix_t* m)   //稀疏矩陣的轉置
{
    matrix_t* out = matrix_new(matrix_row(m), matrix_col(m));
    if (!out) return out;

    double temp;
    for (size_t i = 0; i < matrix_col(m); i++) 
    {
        for (size_t j = 0; j < matrix_row(m); j++) 
        {
            temp = matrix_at(m, i, j);
            if (fabs(temp) > 0.000001)
                matrix_set_at(out, j, i, temp);
        }
    }
    return out;
}

matrix_t* CMat::matrix_add(const matrix_t* m, const matrix_t* n)  //兩矩陣相加
{
    return matrix_add_mm(m, n);
}

matrix_t* CMat::matrix_add_mm(const matrix_t* m, const matrix_t* n)
{
    assert(matrix_col(m) == matrix_col(n));
    assert(matrix_row(m) == matrix_row(n));

    matrix_t* out = matrix_new(matrix_col(m), matrix_row(m));
    if (!out) return out;

    double a, b;
    for (size_t i = 0; i < matrix_col(m); i++) 
    {
        for (size_t j = 0; j < matrix_row(m); j++) 
        {
            a = matrix_at(m, i, j);
            b = matrix_at(n, i, j);
            if (fabs(a + b) < 0.000001) continue;
            matrix_set_at(out, i, j, a + b);
        }
    }
    return out;
}

matrix_t* CMat::matrix_add_ms(const matrix_t* m, double s)    //矩陣和純量相加
{
    matrix_t* out = matrix_new(matrix_col(m), matrix_row(m));
    if (!out) return out;

    double n;
    for (size_t i = 0; i < matrix_col(m); i++) 
    {
        for (size_t j = 0; j < matrix_row(m); j++)
        {
            n = matrix_at(m, i, j);
            if (fabs(n + s) < 0.000001) continue;
            matrix_set_at(out, i, j, n + s);
        }
    }
    return out;
}

matrix_t* CMat::matrix_add_sm(double s, const matrix_t* m)
{
    return matrix_add_ms(m, s);
}

matrix_t* CMat::matrix_prod(const matrix_t* m, const matrix_t* n) //稀疏矩陣的內積 (Product)
{
    assert(matrix_col(m) == matrix_row(n));

    matrix_t* out = matrix_new(matrix_row(m), matrix_col(n));
    if (!out) return out;

    double temp;
    for (size_t i = 0; i < matrix_row(m); i++)
    {
        for (size_t j = 0; j < matrix_col(n); j++) 
        {
            temp = 0.0;
            for (size_t k = 0; k < matrix_col(m); k++)
                temp += matrix_at(m, k, i) * matrix_at(n, j, k);

            if (fabs(temp) > 0.000001)
                matrix_set_at(out, j, i, temp);
        }
    }
    return out;
}