#include "ndarray.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#define M_PI 3.14159265358979323846

// Check 2 ndarrays are compatible for an operation.
#define CHECK_COMPATIBLE(a, b) \
    do { if ((a)->ndim != (b)->ndim) { \
        fprintf(stderr, "ndarray ndim mismatch, ndim : %d != %d\n", (a)->ndim, (b)->ndim); exit(1); \
        } \
        for (int i = 0; i < (a)->ndim; i++) { \
            if ((a)->shape[i] != (b)->shape[i]) { \
                fprintf(stderr, "ndarray shape mismatch, shape[%d] : %d != %d\n", i, (a)->shape[i], (b)->shape[i]); exit(1); \
            } } } while (0)

// Check if the ndarray is a matrix.
#define CHECK_MATRIX(a) \
    do { if ((a)->ndim != 2) { \
        fprintf(stderr, "not a matrix\n"); exit(1); \
        } } while (0)

#define CHECK_MALLOC(a) \
    do { if ((a) == NULL) { \
        fprintf(stderr, "malloc failed\n"); exit(1); \
        } } while (0)

// Create a new ndarray with the given shape.
ndarray *nda_zero(int ndim, int *shape) {
    ndarray *arr = malloc(sizeof(ndarray));
    arr->ndim = ndim;
    arr->shape = malloc(ndim * sizeof(int));
    memcpy(arr->shape, shape, ndim * sizeof(int));
    arr->strides = malloc(ndim * sizeof(int));
    arr->strides[ndim - 1] = 1;
    arr->size = shape[ndim - 1];
    for (int i = ndim - 1; i > 0; i--) {
        arr->strides[i - 1] = arr->strides[i] * arr->shape[i];
        arr->size *= arr->shape[i - 1];
    }
    arr->data = calloc(arr->size, sizeof(float));
    CHECK_MALLOC(arr->data);
    return arr;
}

void nda_init_data(ndarray *arr, float *data){
    memcpy(arr->data, data, arr->size * sizeof(float));
}

void nda_init_rand(ndarray *arr){
    // Initialize the random number generator, range [0, 1]
    for (int i = 0; i < arr->size; i++) {
        arr->data[i] = (float)rand() / (float)RAND_MAX;
    }
}

void nda_print_mat(ndarray* a){
    CHECK_MATRIX(a);
    
    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < a->shape[1]; j++) {
            printf("%f ", a->data[i * a->strides[0] + j * a->strides[1]]);
        }
        printf("\n");
    }
}
void nda_print_shape(ndarray *arr){
    printf("ndarray ndim: %d, size: %d\n", arr->ndim, arr->size);
    printf("ndarray shape: (");
    for (int i = 0; i < arr->ndim; i++) {
        printf("%d", arr->shape[i]);
        if (i < arr->ndim - 1) {
            printf(", ");
        }
    }
    printf(")\n");
    printf("ndarray strides: (");
    for (int i = 0; i < arr->ndim; i++) {
        printf("%d", arr->strides[i]);
        if (i < arr->ndim - 1) {
            printf(", ");
        }
    }
    printf(")\n");
}

// Generate a random number following standard normal distribution
static float randn() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2 * logf(u1)) * cosf(2 * M_PI * u2);
}

void initialize_weights(ndarray *arr) {
    // Initialize the weights of Dense layer or Conv layer.
    float scale; // He initialization scale factor
    if (arr->ndim == 2){
        scale = sqrtf(2. / arr->shape[0]); 
    } else if (arr->ndim == 4){
        scale = sqrtf(2. / (arr->shape[1] * arr->shape[2] * arr->shape[3]));
    } else {
        fprintf(stderr, "ndarray ndim mismatch, ndim : %d\n", arr->ndim); exit(1);
    }

    for (int i = 0; i < arr->size; i++) {
        arr->data[i] = scale * randn();
    }
}

// Free the memory allocated for the ndarray.
void nda_free(ndarray *arr) {
    free(arr->shape);
    free(arr->strides);
    free(arr->data);
    free(arr);
}

// Basic calculations.
#define DEFINE_OP(OP_NAME, OP) \
    void OP_NAME(ndarray *a, ndarray *b, ndarray *out) { \
        CHECK_COMPATIBLE(a, b); \
        CHECK_COMPATIBLE(a, out); \
        for (int i = 0; i < a->size; i++) { \
            out->data[i] = a->data[i] OP b->data[i]; \
        } \
    }

DEFINE_OP(nda_add, +)
DEFINE_OP(nda_sub, -)
DEFINE_OP(nda_mul, *)
DEFINE_OP(nda_div, /)

#define DEFINE_SCA_OP(OP_NAME, OP) \
    void OP_NAME(ndarray *a, float b, ndarray *out) { \
        CHECK_COMPATIBLE(a, out); \
        for (int i = 0; i < a->size; i++) { \
            out->data[i] = a->data[i] OP b; \
        } \
    }

DEFINE_SCA_OP(nda_add_scalar, +)
DEFINE_SCA_OP(nda_sub_scalar, -)
DEFINE_SCA_OP(nda_mul_scalar, *)
DEFINE_SCA_OP(nda_div_scalar, /)

float nda_sum(ndarray *a){
    float sum = 0;
    for (int i = 0; i < a->size; i++) {
        sum += a->data[i];
    }
    return sum;
}

float nda_max(ndarray *a){
    float max = a->data[0];
    for (int i = 1; i < a->size; i++) {
        if (a->data[i] > max) {
            max = a->data[i];
        }
    }
    return max;
}

int nda_argmax(ndarray *a){
    float max = a->data[0];
    int argmax = 0;
    for (int i = 1; i < a->size; i++) {
        if (a->data[i] > max) {
            max = a->data[i];
            argmax = i;
        }
    }
    return argmax;
}

void nda_normalize(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    float sum = nda_sum(a) + 1e-7;
    // Check if sum is zero.
    if (sum == 0) {
        fprintf(stderr, "sum of ndarray is zero\n");
        exit(1);
    }

    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] / sum;
        if (isnan(out->data[i])) {
            fprintf(stderr, "nan in ndarray\n");
            exit(1);
        }
    }
}

// Matrix operations.
void nda_dot(ndarray *a, ndarray *b, ndarray *out){
    CHECK_MATRIX(a);
    CHECK_MATRIX(b);
    // Check shapes.
    if (a->shape[1] != b->shape[0] || out->shape[0] != a->shape[0] || out->shape[1] != b->shape[1]) {
        fprintf(stderr, "ndarray shape mismatch for dot product\n");
        exit(1);
    }

    for (int i = 0; i < a->shape[0]; i++) {
        for (int j = 0; j < b->shape[1]; j++) {
            float sum = 0;
            for (int k = 0; k < a->shape[1]; k++) {
                sum += a->data[i * a->strides[0] + k * a->strides[1]] * b->data[k * b->strides[0] + j * b->strides[1]];
            }
            out->data[i * out->strides[0] + j * out->strides[1]] = sum;
        }
    }
}

void nda_T(ndarray *a){
    CHECK_MATRIX(a);
    int tmp = a->shape[0];
    a->shape[0] = a->shape[1];
    a->shape[1] = tmp;
    tmp = a->strides[0];
    a->strides[0] = a->strides[1];
    a->strides[1] = tmp;
}

void nda_flip(ndarray *mat) {
    // This flip does not consider srtides
    CHECK_MATRIX(mat);

    int rows = mat->shape[0];
    int cols = mat->shape[1];

    for (int i = 0; i < rows / 2; i++){
        for (int j = 0; j < cols; j++){
            float temp = mat->data[i * cols + j];
            mat->data[i * cols + j] = mat->data[(rows - i - 1) * cols + (cols - j - 1)];
            mat->data[(rows - i - 1) * cols + (cols - j - 1)] = temp;
        }
    }

    // If the matrix has an odd number of rows, flip the middle row
    if (rows % 2 == 1) {
        int i = rows / 2;
        for (int j = 0; j < cols / 2; j++) {
            float temp = mat->data[i * cols + j];
            mat->data[i * cols + j] = mat->data[i * cols + (cols - j - 1)];
            mat->data[i * cols + (cols - j - 1)] = temp;
        }
    }
}

void nda_pad(ndarray *a, int pad, ndarray *out){
    // This pad does not guarantee the other part of the matrix is zero
    CHECK_MATRIX(a);
    // Check shapes.
    if (out->shape[0] != a->shape[0] + 2*pad || out->shape[1] != a->shape[1] + 2*pad) {
        fprintf(stderr, "ndarray shape mismatch for padding\n");
        exit(1);
    }
    
    for (int i = 0; i < a->shape[0]; ++i) {
        memcpy(out->data + (i+pad)*out->strides[0] + pad*out->strides[1], 
               a->data + i*a->strides[0], 
               a->shape[1] * sizeof(float));
    }
}

// Ndarray operations.
void nda_reshape(ndarray *a, int ndim, int *shape){
    int* new_stride = malloc(ndim * sizeof(int));
    int* new_shape = malloc(ndim * sizeof(int));
    new_stride[ndim - 1] = 1;
    for (int i = ndim - 1; i > 0; i--) {
        new_stride[i - 1] = new_stride[i] * shape[i];
    }
    if (new_stride[0] * shape[0] != a->strides[0] * a->shape[0]) {
        fprintf(stderr, "ndarray shape mismatch for reshape, size = %d, new size = %d\n", new_stride[0] * shape[0], a->strides[0] * a->shape[0]);
        exit(1);
    }

    a->ndim = ndim;
    free(a->shape);
    free(a->strides);
    a->shape = new_shape;
    memcpy(a->shape, shape, ndim * sizeof(int));
    a->strides = new_stride;
}

ndarray* nda_deepcopy(ndarray *a){
    ndarray* out = malloc(sizeof(ndarray));
    out->ndim = a->ndim;
    out->size = a->size;
    out->shape = malloc(a->ndim * sizeof(int));
    out->strides = malloc(a->ndim * sizeof(int));
    memcpy(out->shape, a->shape, a->ndim * sizeof(int));
    memcpy(out->strides, a->strides, a->ndim * sizeof(int));
    out->data = malloc(a->size * sizeof(float));
    memcpy(out->data, a->data, a->size * sizeof(float));
    return out;
}

void nda_copy(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    memcpy(out->data, a->data, a->size * sizeof(float));
}

void nda_stack(ndarray *a[], int n, ndarray *out){
    // Check shapes.
    for (int i = 1; i < n; i++) {
        CHECK_COMPATIBLE(a[0], a[i]);
    }
    CHECK_COMPATIBLE(a[0], out);
    
    for (int i = 0; i < n; i++) {
        memcpy(out->data + i * a[i]->size, a[i]->data, a[i]->size * sizeof(float));
    }
}

// Convolution operations.
void nda_conv2d(ndarray *a, ndarray *b, ndarray *out){
    CHECK_MATRIX(a);
    CHECK_MATRIX(b);
    CHECK_MATRIX(out);
    // Check shapes.
    if (out->shape[0] != a->shape[0] - b->shape[0] + 1 || out->shape[1] != a->shape[1] - b->shape[1] + 1) {
        fprintf(stderr, "ndarray shape mismatch for conv2d\n");
        exit(1);
    }

    for (int i = 0; i < out->shape[0]; i++) {
        for (int j = 0; j < out->shape[1]; j++) {
            float sum = 0;
            for (int k = 0; k < b->shape[0]; k++) {
                for (int l = 0; l < b->shape[1]; l++) {
                    sum += a->data[(i + k) * a->strides[0] + (j + l) * a->strides[1]] * 
                           b->data[k * b->strides[0] + l * b->strides[1]];
                }
            }
            out->data[i * out->strides[0] + j * out->strides[1]] = sum;
        }
    }
}

void nda_conv3d(ndarray *a, ndarray *b, ndarray *out){
    /* a : 3D, (in_depth, in_height, in_width)
       b : 4D, (filter_num, in_depth, filter_height, filter_width)
       out : 3D, (filter_num, out_height, out_width)
    */
    // Check shapes.
    if (a->ndim != 3 || b->ndim != 4 || out->ndim != 3
        || a->shape[0] != b->shape[1] || out->shape[0] != b->shape[0]
        || out->shape[1] != a->shape[1] - b->shape[2] + 1 
        || out->shape[2] != a->shape[2] - b->shape[3] + 1) {
        fprintf(stderr, "ndarray shape mismatch for conv3d\n");
        exit(1);
    }
    // temp matrix for 2d convolution
    ndarray *mat = nda_zero(2, (int[]){a->shape[1], a->shape[2]});
    ndarray *filter = nda_zero(3, (int[]){b->shape[1], b->shape[2], b->shape[3]});
    ndarray *filter_mat = nda_zero(2, (int[]){b->shape[2], b->shape[3]});
    ndarray *tmp = nda_zero(2, (int[]){out->shape[1], out->shape[2]});
    ndarray *tmp_out = nda_zero(2, (int[]){out->shape[1], out->shape[2]});
    
    for (int i = 0; i < b->shape[0]; i++){
        memcpy(filter->data, b->data + i * b->strides[0], b->strides[0] * sizeof(float));
        for (int j = 0; j < a->shape[0]; j++){
            memcpy(mat->data, a->data + j * a->strides[0], a->strides[0] * sizeof(float));
            memcpy(filter_mat->data, filter->data + j * filter->strides[0], filter->strides[0] * sizeof(float));
            nda_conv2d(mat, filter_mat, tmp);
            nda_add(tmp_out, tmp, tmp_out);
            // printf("---tmp_out---\n");
            // nda_print(tmp_out);
            // printf("------\n");
        }
        memcpy(out->data + i * out->strides[0], tmp_out->data, tmp_out->size * sizeof(float));
    }

    nda_free(mat);
    nda_free(filter);
    nda_free(filter_mat);
    nda_free(tmp);
    nda_free(tmp_out);
}

// Activation functions.
void nda_relu(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] > 0 ? a->data[i] : 0;
    }
}

void nda_identity(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    memcpy(out->data, a->data, a->size * sizeof(float));
}

void nda_softmax(ndarray *a, ndarray *out) {
    CHECK_COMPATIBLE(a, out);
    float max = nda_max(a);

    for (int i = 0; i < a->size; i++) {
        out->data[i] = exp(a->data[i] - max);
        if (isnan(out->data[i])) {
            fprintf(stderr, "nan in ndarray softmax, a->data[%d] = %f, max = %f\n", i, a->data[i], max);
            exit(1);
        }
    }

    nda_normalize(out, out);
}

// Activation function derivatives.
void nda_relu_prime(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    for (int i = 0; i < a->size; i++) {
        out->data[i] = a->data[i] > 0 ? 1 : 0;
    }
}

void nda_identity_prime(ndarray *a, ndarray *out){
    CHECK_COMPATIBLE(a, out);
    for (int i = 0; i < a->size; i++) {
        out->data[i] = 1;
    }
}

// Loss functions.
float mse(ndarray *pr, ndarray *tr){
    CHECK_COMPATIBLE(pr, tr);
    float sum = 0;
    for (int i = 0; i < pr->size; i++) {
        sum += pow(pr->data[i] - tr->data[i], 2);
    }
    return sum / pr->size;
}

void mse_prime(ndarray *pr, ndarray *tr, ndarray *out){
    CHECK_COMPATIBLE(pr, tr);
    CHECK_COMPATIBLE(pr, out);
    for (int i = 0; i < pr->size; i++) {
        out->data[i] = 2 * (pr->data[i] - tr->data[i]);
    }
}

// Cross-entropy loss function
float cross_entropy(ndarray *pr, ndarray *tr){
    CHECK_COMPATIBLE(pr, tr);
    float sum = 0;
    for (int i = 0; i < pr->size; i++) {
        if (pr->data[i] <= 1e-8) {
            sum -= tr->data[i] * log(1e-8);
        } else if (pr->data[i] >= 1 - 1e-8) {
            sum -= tr->data[i] * log(1 - 1e-8);
        } else {
            sum -= tr->data[i] * log(pr->data[i]);
        }
    }
    return sum / pr->size;
}

void cross_entropy_prime(ndarray *pr, ndarray *tr, ndarray *out){
    CHECK_COMPATIBLE(pr, tr);
    CHECK_COMPATIBLE(pr, out);
    for (int i = 0; i < pr->size; i++) {
        out->data[i] = pr->data[i] - tr->data[i];
    }
}

// Optimizers.
void sgd(ndarray *w, ndarray *dw, float lr){
    CHECK_COMPATIBLE(w, dw);
    for (int i = 0; i < w->size; i++) {
        w->data[i] -= lr * dw->data[i];
    }
}
