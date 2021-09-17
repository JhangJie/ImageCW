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
    matrix_t* matrix_trans(const matrix_t* m);   //�}���x�}����m

public:
    CMat();
    ~CMat();

    void matrix_delete(void* self);
    matrix_t* matrix_new(size_t col, size_t row);

    bool matrix_is_equal(const matrix_t* m, const matrix_t* n);     //�T�{��x�}�۵�

    size_t matrix_col(const matrix_t* self);    //���o�}���x�}���j�p (�κ���)
    size_t matrix_row(const matrix_t* self);

    double matrix_at(const matrix_t* self, size_t col, size_t row);              //���o�}���x�}�����N��m����
    bool matrix_set_at(matrix_t* self, size_t col, size_t row, double value);    //�s�J�}���x�}�����N��m����

    //�s�J�}���x�}�����N��m����
    void matrix_delete_at(matrix_t* self, size_t col, size_t row);
    bool matrix_expand(matrix_t* self);

    //��x�}�ۥ[
    matrix_t* matrix_add(const matrix_t* m, const matrix_t* n);
    matrix_t* matrix_add_mm(const matrix_t* m, const matrix_t* n);

    //�x�}�M�¶q�ۥ[
    matrix_t* matrix_add_ms(const matrix_t* m, double s);
    matrix_t* matrix_add_sm(double s, const matrix_t* m);

    matrix_t* matrix_prod(const matrix_t* m, const matrix_t* n);    //�}���x�}�����n (Product)
};
