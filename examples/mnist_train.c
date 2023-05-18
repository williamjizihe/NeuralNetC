#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "network.h"
#include "ndarray.h"
#include "misc.h"

#define IMAGE_SIZE 20
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

float valuate(Network* network, ndarray** images, int* labels, int num) {
    int correct = 0; 
    ndarray* output = nda_zero(2, (int[]){10, 1});
    for (int i = 0; i < num; i++) {
        network_forward(network, images[i], output);
        correct += nda_argmax(output) == labels[i];
    }
    nda_free(output);
    return (float)correct / num;
}

int main() {
    srand(time(NULL));  

    time_t current_time;
    time(&current_time);
    struct tm *local_time = localtime(&current_time);

    char logname[50], networkname[50];
    sprintf(logname, "../logs/log_%d_%d_%d_%d_%d_%d.txt", 
            local_time->tm_year+1900, local_time->tm_mon+1, local_time->tm_mday,
            local_time->tm_hour, local_time->tm_min, local_time->tm_sec);
    sprintf(networkname, "../models/network_%d_%d_%d_%d_%d_%d.txt", 
            local_time->tm_year+1900, local_time->tm_mon+1, local_time->tm_mday,
            local_time->tm_hour, local_time->tm_min, local_time->tm_sec);
    FILE *file = fopen(logname, "w");
    if (file == NULL) {
        printf("Failed to open the file.\n");
    }

    int train_num = 3500;
    int val_num = 750;

    ndarray** train_images = (ndarray**)(malloc(train_num * sizeof(ndarray*)));
    int* train_labels = (int*)(malloc(train_num * sizeof(int)));
    
    ndarray** val_images = (ndarray**)(malloc(val_num * sizeof(ndarray*)));
    int* val_labels = (int*)(malloc(val_num * sizeof(int)));

    read_data("../datasets/mnist_20x20/train_labels.txt", train_images, train_labels, train_num, IMAGE_SIZE);
    read_data("../datasets/mnist_20x20/val_labels.txt", val_images, val_labels, val_num, IMAGE_SIZE);

    data_shuffle(train_images, train_labels, train_num);

    // Initialize the network
    Network* network = create_network(0.003);
    ndarray* target = nda_zero(2, (int[]){10, 1});
    ndarray* output = nda_zero(2, (int[]){10, 1});

    float best_val_acc = 0.0;
    Network* best_network = create_network(0.003);

    for(int epoch = 1; epoch <= 100; epoch++) {
        float loss = 0;
        int correct = 0;

        for(int i = 0; i < train_num; i++) {
            // Forward
            network_forward(network, train_images[i], output);
            // Check the prediction
            correct += nda_argmax(output) == train_labels[i];
            // Backward
            memset(target->data, 0, 10 * sizeof(float));
            target->data[train_labels[i]] = 1.0;
            network_backward(network, target);
            // Update
            network_update(network);
            // Accumulate loss
            loss += network->loss;
        }
        float val_acc = valuate(network, val_images, val_labels, val_num);
        float train_acc = (float)correct / train_num;
        // Print the loss
        printf("Epoch %d: loss = %f, train acc = %.2f%%, val acc = %.2f%%, learning rate = %f\n", 
                epoch, loss / train_num, train_acc * 100, val_acc * 100, network->learning_rate);
        fprintf(file, "Epoch %d: loss = %f, train acc = %f, val acc = %f, learning rate = %f\n", 
                epoch, loss / train_num, train_acc * 100, val_acc * 100, network->learning_rate);
        // Update learning rate
        if (train_acc > 0.2){
            network->learning_rate  = MAX(0.0003, network->learning_rate * 0.99);
        }
        // Save the best network
        if (val_acc > best_val_acc) {
            best_val_acc = val_acc;
            copy_network(network, best_network);
        }
    }

    // Close the file
    fclose(file);
    printf("Training finished. Save network\n");
    save_network(best_network, networkname);

    // Free the memory
    for(int i = 0; i < train_num; i++) {
        nda_free(train_images[i]);
    }
    for(int i = 0; i < val_num; i++) {
        nda_free(val_images[i]);
    }
    free(train_images), free(train_labels);
    free(val_images), free(val_labels);
    nda_free(target), nda_free(output);
    free_network(network);
    return 0;
}