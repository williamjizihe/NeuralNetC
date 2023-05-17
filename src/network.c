#include "network.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Network *create_network(float learning_rate){
    Network *network = malloc(sizeof(Network));
    network->dense1 = create_dense_layer(RELU);
    network->dense2 = create_dense_layer(RELU);
    network->dense3 = create_dense_layer(SOFTMAX);

    network->d1_output = nda_zero(2, (size_t[]){256, 1});
    network->d1_input_grad = nda_zero(2, (size_t[]){256, 1});

    network->d2_output = nda_zero(2, (size_t[]){128, 1});
    network->d2_input_grad = nda_zero(2, (size_t[]){128, 1});

    network->d3_output = nda_zero(2, (size_t[]){10, 1});
    network->d3_input_grad = nda_zero(2, (size_t[]){10, 1});

    network->loss = 0;
    network->learning_rate = learning_rate;
    return network;
}

void network_forward(Network *self, ndarray *input, ndarray *output){
    // input : (400, 1)
    // output: (10, 1)
    self->dense1->forward(self->dense1, input, self->d1_output);
    for (size_t i = 0; i < 10; i++){
        if (isnan(self->d1_output->data[i])){
            fprintf(stderr, "NaN in d1_output\n"); exit(1);
        }
    }
    self->dense2->forward(self->dense2, self->d1_output, self->d2_output);
    for (size_t i = 0; i < 10; i++){
        if (isnan(self->d2_output->data[i])){
            fprintf(stderr, "NaN in d2_output\n"); exit(1);
        }
    }
    self->dense3->forward(self->dense3, self->d2_output, self->d3_output);
    nda_copy(self->d3_output, output);
    // Check if there is NaN in output
    for (size_t i = 0; i < 10; i++){
        if (isnan(output->data[i])){
            fprintf(stderr, "NaN in output\n"); exit(1);
        }
    }
}

void network_backward(Network *self, ndarray *target){
    // target: (10, 1)
    self->loss = cross_entropy(self->d3_output, target);
    cross_entropy_prime(self->d3_output, target, self->d3_input_grad);
    self->dense3->backward(self->dense3, self->d3_input_grad, self->d2_input_grad);
    self->dense2->backward(self->dense2, self->d2_input_grad, self->d1_input_grad);
    self->dense1->backward(self->dense1, self->d1_input_grad, NULL);
}

void network_update(Network *self){
    sgd(self->dense1->bias, self->dense1->bias_grad, self->learning_rate);
    sgd(self->dense1->weights, self->dense1->weights_grad, self->learning_rate);
    sgd(self->dense2->bias, self->dense2->bias_grad, self->learning_rate);
    sgd(self->dense2->weights, self->dense2->weights_grad, self->learning_rate);
    sgd(self->dense3->bias, self->dense3->bias_grad, self->learning_rate);
    sgd(self->dense3->weights, self->dense3->weights_grad, self->learning_rate);
}

void free_network(Network *self){
    free_dense_layer(self->dense1);
    free_dense_layer(self->dense2);
    free_dense_layer(self->dense3);

    nda_free(self->d1_output);
    nda_free(self->d1_input_grad);

    nda_free(self->d2_output);
    nda_free(self->d2_input_grad);
    
    nda_free(self->d3_output);
    nda_free(self->d3_input_grad);
    free(self);
}