#include "cnn.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

CNN *create_network(float learning_rate){
    // input: (1, 20, 20)
    CNN *network = malloc(sizeof(CNN));

    network->conv1 = create_conv_layer(32, 3, RELU);
    network->conv2 = create_conv_layer(64, 3, RELU);
    network->flat1 = create_flatten_layer();
    network->dense1 = create_dense_layer(RELU);
    network->dense2 = create_dense_layer(SOFTMAX);

    network->c1_output = nda_zero(3, (int[]){32, 18, 18});
    network->c1_input_grad = nda_zero(3, (int[]){32, 18, 18});

    network->c2_output = nda_zero(3, (int[]){64, 16, 16});
    network->c2_input_grad = nda_zero(3, (int[]){64, 16, 16});

    network->f1_output = nda_zero(2, (int[]){64*16*16, 1});
    network->f1_input_grad = nda_zero(2, (int[]){64*16*16, 1});

    network->d1_output = nda_zero(2, (int[]){128, 1});
    network->d1_input_grad = nda_zero(2, (int[]){128, 1});

    network->d2_output = nda_zero(2, (int[]){10, 1});
    network->d2_input_grad = nda_zero(2, (int[]){10, 1});

    network->loss = 0;
    network->learning_rate = learning_rate;
    return network;
}

void network_forward(CNN *self, ndarray *input, ndarray *output){
    // input : (1, 20, 20)
    // output: (10, 1)
    self->conv1->forward(self->conv1, input, self->c1_output);
    self->conv2->forward(self->conv2, self->c1_output, self->c2_output);
    self->flat1->forward(self->c2_output, self->f1_output);
    self->dense1->forward(self->dense1, self->f1_output, self->d1_output);
    self->dense2->forward(self->dense2, self->d1_output, output);
    nda_copy(self->d2_output, output);
}

void network_backward(CNN *self, ndarray *target){
    // target: (10, 1)
    self->loss = cross_entropy(self->d2_output, target);
    cross_entropy_prime(self->d2_output, target, self->d2_input_grad);
    self->dense2->backward(self->dense2, self->d2_input_grad, self->d1_input_grad);
    self->dense1->backward(self->dense1, self->d1_input_grad, self->f1_input_grad);
    self->flat1->backward(self->f1_input_grad, self->c2_input_grad);
    self->conv2->backward(self->conv2, self->c2_input_grad, self->c1_input_grad);
    self->conv1->backward(self->conv1, self->c1_input_grad, NULL);
}

void network_update(CNN *self){
    sgd(self->conv1->weights, self->conv1->weights_grad, self->learning_rate);
    sgd(self->conv1->bias, self->conv1->bias_grad, self->learning_rate);
    // printf("conv1\n");
    sgd(self->conv2->weights, self->conv2->weights_grad, self->learning_rate);
    sgd(self->conv2->bias, self->conv2->bias_grad, self->learning_rate);

    sgd(self->dense1->weights, self->dense1->weights_grad, self->learning_rate);
    sgd(self->dense1->bias, self->dense1->bias_grad, self->learning_rate);
    // printf("dense1\n");
    sgd(self->dense2->weights, self->dense2->weights_grad, self->learning_rate);
    sgd(self->dense2->bias, self->dense2->bias_grad, self->learning_rate);
    // printf("dense2\n");
}

void free_network(CNN *self){
    free_dense_layer(self->dense1);
    free_dense_layer(self->dense2);
    free_flatten_layer(self->flat1);
    free_conv_layer(self->conv1);
    free_conv_layer(self->conv2);

    nda_free(self->d1_output);
    nda_free(self->d1_input_grad);

    nda_free(self->d2_output);
    nda_free(self->d2_input_grad);
    
    nda_free(self->f1_output);
    nda_free(self->f1_input_grad);

    nda_free(self->c1_output);
    nda_free(self->c1_input_grad);

    nda_free(self->c2_output);
    nda_free(self->c2_input_grad);
    
    free(self);
}