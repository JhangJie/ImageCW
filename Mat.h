typedef struct
{
    size_t col;
    size_t row;
    size_t size;
    size_t capacity;
    size_t* cols;
    size_t* rows;
    double* elements;
}matrix_t;

class CMat
{
private:
    matrix_t* matrix_trans(const matrix_t* m);   //稀疏矩陣的轉置

public:
    CMat();
    ~CMat();

    void matrix_delete(void* self);
    matrix_t* matrix_new(size_t col, size_t row);

    bool matrix_is_equal(const matrix_t* m, const matrix_t* n);     //確認兩矩陣相等

    size_t matrix_col(const matrix_t* self);    //取得稀疏矩陣的大小 (或維度)
    size_t matrix_row(const matrix_t* self);

    double matrix_at(const matrix_t* self, size_t col, size_t row);              //取得稀疏矩陣中任意位置的值
    bool matrix_set_at(matrix_t* self, size_t col, size_t row, double value);    //存入稀疏矩陣中任意位置的值

    //存入稀疏矩陣中任意位置的值
    void matrix_delete_at(matrix_t* self, size_t col, size_t row);
    bool matrix_expand(matrix_t* self);

    //兩矩陣相加
    matrix_t* matrix_add(const matrix_t* m, const matrix_t* n);
    matrix_t* matrix_add_mm(const matrix_t* m, const matrix_t* n);

    //矩陣和純量相加
    matrix_t* matrix_add_ms(const matrix_t* m, double s);
    matrix_t* matrix_add_sm(double s, const matrix_t* m);

    matrix_t* matrix_prod(const matrix_t* m, const matrix_t* n);    //稀疏矩陣的內積 (Product)
};
