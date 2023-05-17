#ifndef NETWORK_H
#define NETWORK_H

#include "layer.h"

typedef struct network
{
    float learning_rate;
    float loss;

    DenseLayer *dense1;
    DenseLayer *dense2;
    DenseLayer *dense3;

    ndarray *d1_output;
    ndarray *d1_input_grad;
    
    ndarray *d2_output;
    ndarray *d2_input_grad;

    ndarray *d3_output;
    ndarray *d3_input_grad;
} Network;

Network *create_network(float learning_rate);
void network_forward(Network *self, ndarray *input, ndarray *output);
void network_backward(Network *self, ndarray *target);
void network_update(Network *self);
void free_network(Network *self);

#endif // NETWORK_H
