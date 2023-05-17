#ifndef LAYER_H
#define LAYER_H

#include "ndarray.h"

typedef enum {
    NONE,
    RELU,
    SOFTMAX,
} ActivationType;

typedef struct denselayer
{
    ActivationType activation;
    ndarray *input;
    ndarray *weights;
    ndarray *bias;
    ndarray *weights_grad;
    ndarray *bias_grad;
    ndarray *linear_output;
    void (*forward)(struct denselayer *self, ndarray *input, ndarray *output);
    void (*backward)(struct denselayer *self, ndarray *input_grad, ndarray *output_grad); 
} DenseLayer;

typedef struct convlayer
{
    ActivationType activation;
    size_t kernel_num;
    size_t kernel_size;
    ndarray *input;
    ndarray *weights;
    ndarray *bias;
    ndarray *weights_grad;
    ndarray *bias_grad;
    ndarray *linear_output;
    void (*forward)(struct convlayer *self, ndarray *input, ndarray *output);
    void (*backward)(struct denselayer *self, ndarray *input_grad, ndarray *output_grad);
} ConvLayer;

// function prototypes for creating layers
DenseLayer *create_dense_layer(ActivationType activation);
ConvLayer *create_conv_layer(size_t kernel_num, size_t kernel_size, ActivationType activation);

void free_dense_layer(DenseLayer *layer);
void free_conv_layer(ConvLayer *layer);

#endif // LAYER_H
