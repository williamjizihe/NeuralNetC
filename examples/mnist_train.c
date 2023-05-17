#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "network.h"
#include "ndarray.h"
#include "misc.h"

#define IMAGE_SIZE 20
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

void read_data(const char* filename, ndarray** images, int* labels, int num) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("Cannot open file %s\n", filename);
        exit(1);
    }

    // Read each line from the file
    for (int i = 0; i < num; i++) {
        char image_file[255];
        int label;
        fscanf(file, "%s %d", image_file, &label);

        labels[i] = label;
        images[i] = nda_zero(2, (size_t[]){IMAGE_SIZE * IMAGE_SIZE, 1});

        // Open the image file and read the data
        FILE* image_file_handle = fopen(image_file, "r");
        if (image_file_handle == NULL) {
            printf("Cannot open file %s\n", image_file);
            exit(1);
        }
        
        for (int j = 0; j < IMAGE_SIZE * IMAGE_SIZE; j++) {
            int pixel_value;
            fscanf(image_file_handle, "%d", &pixel_value);
            images[i]->data[j] = (float)pixel_value;
        }
        fclose(image_file_handle);
    }

    fclose(file);
}

float valuate(Network* network, ndarray** images, int* labels, int num) {
    int correct = 0; 
    ndarray* output = nda_zero(2, (size_t[]){10, 1});
    for (int i = 0; i < num; i++) {
        network_forward(network, images[i], output);
        correct += (int)nda_argmax(output) == labels[i];
    }
    nda_free(output);
    return (float)correct / num;
}

int main() {
    srand(time(NULL));  
    int test_num = 500;
    int val_num = 50;

    ndarray** test_images = (ndarray**)(malloc(test_num * sizeof(ndarray*)));
    int* test_labels = (int*)(malloc(test_num * sizeof(int)));
    
    ndarray** val_images = (ndarray**)(malloc(val_num * sizeof(ndarray*)));
    int* val_labels = (int*)(malloc(val_num * sizeof(int)));

    read_data("../datasets/mnist_20x20/train_labels.txt", test_images, test_labels, test_num);
    read_data("../datasets/mnist_20x20/val_labels.txt", val_images, val_labels, val_num);

    data_shuffle(test_images, test_labels, test_num);

    // Print the first 10 image
    for (int n = 0; n < 10; n++){
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                printf("%d ", (int)test_images[n]->data[i * IMAGE_SIZE + j]);
            }
            printf("\n");
        }
        printf("label : %d\n", test_labels[n]);
    }
    
    // Initialize the network
    Network* network = create_network(0.01);
    ndarray* target = nda_zero(2, (size_t[]){10, 1});
    ndarray* output = nda_zero(2, (size_t[]){10, 1});

    for(int epoch = 1; epoch <= 300; epoch++) {
        float loss = 0;
        int correct = 0;

        int rand_index = rand() % test_num;
        network_forward(network, test_images[rand_index], output);
        for (int j = 0; j < 10; j++) {
            printf("%f ", output->data[j]);
        }

        printf("target: %d\n", test_labels[rand_index]);
        for(int i = 0; i < test_num; i++) {
            // Forward
            network_forward(network, test_images[i], output);
            // Check the prediction
            correct += (int)nda_argmax(output) == test_labels[i];
            // Backward
            memset(target->data, 0, 10 * sizeof(float));
            target->data[test_labels[i]] = 1.0;
            network_backward(network, target);
            // Update
            network_update(network);
            // Accumulate loss
            loss += network->loss;

            // if (i == 0){
            //     for (int j = 0; j < 10; j++) {
            //         printf("%f ", output->data[j]);
            //     }
            //     printf("\n");
            //     for (int j = 0; j < 10; j++) {
            //         printf("%f ", target->data[j]);
            //     }
            //     printf("\n");
            // }
        }

        // Print the loss
        printf("Epoch %d: loss = %f, train acc = %f, val acc = %f\n", 
                epoch, loss / test_num, (float)correct / test_num, valuate(network, val_images, val_labels, val_num));
        // Update learning rate
        // network->learning_rate  = MAX(0.0001, network->learning_rate * 0.9);
    }
    // printf("Final weight:\n");
    // nda_print_mat(network->dense1->weights);
    // printf("Final bias:\n");
    // nda_print_mat(network->dense1->bias);
    // Free the memory
    for(int i = 0; i < test_num; i++) {
        nda_free(test_images[i]);
    }
    for(int i = 0; i < val_num; i++) {
        nda_free(val_images[i]);
    }
    free(test_images);
    free(test_labels);
    free(val_images);
    free(val_labels);
    nda_free(target);
    nda_free(output);
    free_network(network);
    return 0;
}