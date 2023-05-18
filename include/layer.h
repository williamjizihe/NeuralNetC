#ifndef LAYER_H
#define LAYER_H

#include "ndarray.h"

#include <stdio.h>

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
    int kernel_num;
    int kernel_size;
    ndarray *input;
    ndarray *weights;
    ndarray *bias;
    ndarray *weights_grad;
    ndarray *bias_grad;
    ndarray *linear_output;
    void (*forward)(struct convlayer *self, ndarray *input, ndarray *output);
    void (*backward)(struct convlayer *self, ndarray *input_grad, ndarray *output_grad);
} ConvLayer;

typedef struct flattenlayer
{
    void (*forward)(ndarray *input, ndarray *output);
    void (*backward)(ndarray *input_grad, ndarray *output_grad);
} FlattenLayer;

// function prototypes for creating layers
DenseLayer *create_dense_layer(ActivationType activation);
ConvLayer *create_conv_layer(int kernel_num, int kernel_size, ActivationType activation);
FlattenLayer *create_flatten_layer();

void save_dense_layer(DenseLayer *layer, FILE *file);
void save_conv_layer(ConvLayer *layer, FILE *file);

void load_dense_layer(DenseLayer *layer, FILE *file);
void load_conv_layer(ConvLayer *layer, FILE *file);

void copy_dense_layer(DenseLayer *dst, DenseLayer *src);
void copy_conv_layer(ConvLayer *dst, ConvLayer *src);

void free_dense_layer(DenseLayer *layer);
void free_conv_layer(ConvLayer *layer);
void free_flatten_layer(FlattenLayer *layer);
#endif // LAYER_H
