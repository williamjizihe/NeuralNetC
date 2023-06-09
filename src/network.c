#include "network.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

Network *create_network(float learning_rate){
    Network *network = malloc(sizeof(Network));
    network->dense1 = create_dense_layer(RELU);
    network->dense2 = create_dense_layer(RELU);
    network->dense3 = create_dense_layer(SOFTMAX);

    network->d1_output = nda_zero(2, (int[]){256, 1});
    network->d1_input_grad = nda_zero(2, (int[]){256, 1});

    network->d2_output = nda_zero(2, (int[]){128, 1});
    network->d2_input_grad = nda_zero(2, (int[]){128, 1});

    network->d3_output = nda_zero(2, (int[]){10, 1});
    network->d3_input_grad = nda_zero(2, (int[]){10, 1});

    network->loss = 0;
    network->learning_rate = learning_rate;
    return network;
}

void network_forward(Network *self, ndarray *input, ndarray *output){
    // input : (400, 1)
    // output: (10, 1)
    self->dense1->forward(self->dense1, input, self->d1_output);
    self->dense2->forward(self->dense2, self->d1_output, self->d2_output);
    self->dense3->forward(self->dense3, self->d2_output, self->d3_output);
    nda_copy(self->d3_output, output);
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

void save_network(Network *network, const char *filename) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // Save parameters for each layer
    save_dense_layer(network->dense1, file);
    save_dense_layer(network->dense2, file);
    save_dense_layer(network->dense3, file);

    fclose(file);
}

void load_network(Network *network, const char *filename) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file!\n");
        exit(1);
    }

    // Load parameters for each layer
    load_dense_layer(network->dense1, file);
    load_dense_layer(network->dense2, file);
    load_dense_layer(network->dense3, file);

    printf("Loaded network from %s\n", filename);
    fclose(file);
}

void copy_network(Network *dst, Network *src){
    copy_dense_layer(dst->dense1, src->dense1);
    copy_dense_layer(dst->dense2, src->dense2);
    copy_dense_layer(dst->dense3, src->dense3);
}