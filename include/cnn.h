#ifndef CNN_H
#define CNN_H
#include "ndarray.h"
#include "layer.h"

typedef struct cnn
{
    float learning_rate;
    float loss;

    ConvLayer *conv1;
    FlattenLayer *flat1;
    DenseLayer *dense1;
    DenseLayer *dense2;

    ndarray *c1_output;
    ndarray *f1_output;
    ndarray *d1_output;
    ndarray *d2_output;

    ndarray *c1_input_grad;
    ndarray *f1_input_grad;
    ndarray *d1_input_grad;
    ndarray *d2_input_grad;
} CNN;

CNN *create_network(float learning_rate);
void network_forward(CNN *self, ndarray *input, ndarray *output);
void network_backward(CNN *self, ndarray *target);
void network_update(CNN *self);
void free_network(CNN *self);

void copy_network(CNN *dst, CNN *src);
void save_network(CNN *network, const char *filename);
void load_network(CNN *network, const char *filename);
#endif // CNN_H