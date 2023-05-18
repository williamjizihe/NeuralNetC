#include "cnn.h"
#include "layer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(time(NULL));

    CNN *network = create_network(1);
    // random input
    ndarray *input1 = nda_zero(3, (int[]){1, 20, 20});
    ndarray *input2 = nda_zero(3, (int[]){1, 20, 20});
    nda_init_rand(input1);
    nda_init_rand(input2);
    // random target
    ndarray *target1 = nda_zero(2, (int[]){10, 1});
    ndarray *target2 = nda_zero(2, (int[]){10, 1});
    target1->data[2] = 1;
    target2->data[3] = 1;

    ndarray *output = nda_zero(2, (int[]){10, 1});

    for (int i = 0; i < 3000; i++){
        network_forward(network, input1, output);
        network_backward(network, target1);
        network_update(network);
        printf("loss: %f, output: ", network->loss);
        for (int j = 0; j < 10; j++){
            printf("%f ", output->data[j]);
        } 
        printf("target: %d\n", nda_argmax(target1));

        printf("Dense2 Weights: ");
        for (int j = 0; j < 15; j++){
            printf("%f ", network->conv1->weights->data[j]);
        }
        printf("\n");

        network_forward(network, input2, output);
        network_backward(network, target2);
        network_update(network);
        printf("loss: %f, output: ", network->loss);
        for (int j = 0; j < 10; j++){
            printf("%f ", output->data[j]);
        } 
        printf("target: %d\n", nda_argmax(target2));
        // update learning rate
        // network->learning_rate *= 0.99;
    }
    free_network(network);
    nda_free(input1);
    nda_free(input2);
    nda_free(target1);
    nda_free(target2);
    nda_free(output);
    return 0;
}