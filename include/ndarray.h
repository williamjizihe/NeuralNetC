#ifndef NDARRAY_H
#define NDARRAY_H

#include <stddef.h>

typedef struct {
    int ndim;
    int size;
    int *shape;
    int *strides;
    float *data;
} ndarray;

// Create a new ndarray with the given shape.
ndarray *nda_zero(int ndim, int *shape);

void nda_init_data(ndarray *arr, float *data);
void nda_init_rand(ndarray *arr);
void initialize_weights(ndarray *arr);

// Print the ndarray.
void nda_print_mat(ndarray *arr);
void nda_print_shape(ndarray *arr);

// Free the memory allocated for the ndarray.
void nda_free(ndarray *arr);

// Basic calculations on ndarrays.
void nda_add(ndarray *a, ndarray *b, ndarray *out);
void nda_sub(ndarray *a, ndarray *b, ndarray *out);
void nda_mul(ndarray *a, ndarray *b, ndarray *out);
void nda_div(ndarray *a, ndarray *b, ndarray *out);
void nda_add_scalar(ndarray *a, float b, ndarray *out);
void nda_sub_scalar(ndarray *a, float b, ndarray *out);
void nda_mul_scalar(ndarray *a, float b, ndarray *out);
void nda_div_scalar(ndarray *a, float b, ndarray *out);
float nda_sum(ndarray *a);
float nda_max(ndarray *a);
int nda_argmax(ndarray *a);
void nda_normalize(ndarray *a, ndarray *out);

// Ndarray operations.
void nda_reshape(ndarray *a, int ndim, int *shape);
ndarray* nda_deepcopy(ndarray *a);
void nda_copy(ndarray *a, ndarray *out);
void nda_stack(ndarray *a[], int n, ndarray *out);

// Matrix operations.
void nda_dot(ndarray *a, ndarray *b, ndarray *out);
void nda_T(ndarray *a);
void nda_flip(ndarray *a);
void nda_pad(ndarray *a, int pad, ndarray *out);

// Convolution operations.
void nda_conv2d(ndarray *a, ndarray *b, ndarray *out);
void nda_conv3d(ndarray *a, ndarray *b, ndarray *out);

// Activation functions.
void nda_relu(ndarray *a, ndarray *out);
void nda_identity(ndarray *a, ndarray *out);
void nda_softmax(ndarray *a, ndarray *out);

// Activation function derivatives.
void nda_relu_prime(ndarray *a, ndarray *out);
void nda_identity_prime(ndarray *a, ndarray *out);

// Loss functions.
float mse(ndarray *pr, ndarray *tr);
void mse_prime(ndarray *pr, ndarray *tr, ndarray *out);

float cross_entropy(ndarray *pr, ndarray *tr);
void cross_entropy_prime(ndarray *pr, ndarray *tr, ndarray *out);

// Optimizers.
void sgd(ndarray *w, ndarray *dw, float lr);

#endif // NDARRAY_H
