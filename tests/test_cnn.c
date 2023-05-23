#include "cnn.h"
#include "layer.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    srand(time(NULL));

    time_t current_time;
    time(&current_time);
    struct tm *local_time = localtime(&current_time);

    char networkname[50];
    sprintf(networkname, "../models/test_%d_%d_%d_%d_%d_%d.txt", 
            local_time->tm_year+1900, local_time->tm_mon+1, local_time->tm_mday,
            local_time->tm_hour, local_time->tm_min, local_time->tm_sec);

    CNN *network = create_network(0.03);
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

    for (int i = 0; i < 100; i++){
        network_forward(network, input1, output);
        network_backward(network, target1);
        network_update(network);
        printf("loss: %f, output: ", network->loss);
        for (int j = 0; j < 10; j++){
            printf("%f ", output->data[j]);
        } 
        printf("target: %d\n", nda_argmax(target1));

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
    // save_network(network, networkname);

    free_network(network);
    nda_free(input1);
    nda_free(input2);
    nda_free(target1);
    nda_free(target2);
    nda_free(output);
    return 0;
}