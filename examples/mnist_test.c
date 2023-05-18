#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "network.h"
#include "ndarray.h"
#include "misc.h"

#define IMAGE_SIZE 20

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

int main(int argc, char* argv[]){
    if (argc < 2){
        printf("Usage: ./mnist_test.x <model_path>\n");
        return 0;
    }
    // Load the network
    Network* network = create_network(0.003);
    load_network(network, argv[1]);

    // Print the test accuracy
    int test_num = 750;
    ndarray** test_images = (ndarray**)(malloc(test_num * sizeof(ndarray*)));
    int* test_labels = (int*)(malloc(test_num * sizeof(int)));
    read_data("../datasets/mnist_20x20/test_labels.txt", test_images, test_labels, test_num, IMAGE_SIZE);
    printf("Test data loaded.\n");

    float test_acc = valuate(network, test_images, test_labels, test_num);
    printf("Test accuracy: %.2f%%\n", test_acc * 100);

    // Free the test data
    for (int i = 0; i < test_num; i++) {
        nda_free(test_images[i]);
    }
    free(test_images);
    free(test_labels);
    
    // Predict the image given by the user
    char image_path[100], img_name[50];
    ndarray* image = nda_zero(2, (int[]){IMAGE_SIZE*IMAGE_SIZE, 1});
    printf("Enter the index of the image: ");
    scanf("%s", img_name);
    sprintf(image_path, "../datasets/mnist_20x20/test/%s.txt", img_name);
    read_image(image_path, image, IMAGE_SIZE);
    ndarray* output = nda_zero(2, (int[]){10, 1});
    network_forward(network, image, output);

    // Print the prediction
    printf("Prediction: %d\n", nda_argmax(output));

    // Free the memory
    nda_free(image);
    nda_free(output);
    free_network(network);
    return 0;
}