#include "layer.h"

#include <string.h>
#include <stdlib.h>
#include <stdio.h>

typedef void (*ActivationFunc)(ndarray *a, ndarray *out);
typedef void (*ActivationDFunc)(ndarray *a, ndarray *out);

ActivationFunc activation_functions[] = {
    [NONE] = nda_identity,
    [RELU] = nda_relu,
    [SOFTMAX] = nda_softmax,
    // Add more activation functions here...
};

ActivationDFunc activation_function_derivatives[] = {
    [NONE] = nda_identity_prime,
    [RELU] = nda_relu_prime,
    [SOFTMAX] = nda_identity_prime, // softmax must be used with cross entropy loss
    // Add more activation function derivatives here...
};

static void dense_forward(DenseLayer *self, ndarray *input, ndarray *output){
    // Initialize weights and bias.
    if (self->weights == NULL) {
        printf("Initializing weights and bias for dense layer\n");
        self->weights = nda_zero(2, (int[]){output->shape[0], input->shape[0]});
        self->bias = nda_zero(2, (int[]){output->shape[0], 1});
        initialize_weights(self->weights);
        nda_init_rand(self->bias);
        nda_div_scalar(self->weights, 100, self->weights);

        printf("Weights shape : "); nda_print_shape(self->weights);
        printf("Bias shape : "); nda_print_shape(self->bias); printf("\n");

        self->weights_grad = nda_zero(2, (int[]){output->shape[0], input->shape[0]});
        self->bias_grad = nda_zero(2, (int[]){output->shape[0], 1});
        self->linear_output = nda_zero(2, (int[]){output->shape[0], 1});
    }
    self->input = input;
    nda_dot(self->weights, input, self->linear_output);
    nda_add(self->linear_output, self->bias, self->linear_output);
    activation_functions[self->activation](self->linear_output, output);
}

static void dense_backward(DenseLayer *self, ndarray *input_grad, ndarray *output_grad){
    activation_function_derivatives[self->activation](self->linear_output, self->linear_output);
    nda_mul(input_grad, self->linear_output, self->bias_grad);
    
    nda_T(self->input);  // Transpose in-place
    nda_dot(input_grad, self->input, self->weights_grad);
    nda_T(self->input);  // Transpose back

    if (output_grad != NULL) {
        nda_T(self->weights);  // Transpose in-place
        nda_dot(self->weights, input_grad, output_grad);
        nda_T(self->weights);  // Transpose back
    }
}

DenseLayer *create_dense_layer(ActivationType activation){
    DenseLayer *layer = malloc(sizeof(DenseLayer));
    layer->activation = activation;
    layer->input = NULL;
    layer->weights = NULL;
    layer->bias = NULL;
    layer->weights_grad = NULL;
    layer->bias_grad = NULL;
    layer->forward = dense_forward;
    layer->backward = dense_backward;
    return layer;
}

void free_dense_layer(DenseLayer *layer){
    if (layer->weights != NULL) nda_free(layer->weights);
    if (layer->bias != NULL) nda_free(layer->bias);
    if (layer->weights_grad != NULL) nda_free(layer->weights_grad);
    if (layer->bias_grad != NULL) nda_free(layer->bias_grad);
    if (layer->linear_output != NULL) nda_free(layer->linear_output);
    free(layer);
}

static void conv_forward(ConvLayer *self, ndarray *input, ndarray *output){
    // Initialize weights and bias.
    if (self->weights == NULL) {
        printf("Initializing weights and bias for conv layer\n");
        self->weights = nda_zero(4, (int[]){self->kernel_num, input->shape[0], self->kernel_size, self->kernel_size});
        self->bias = nda_zero(3, (int[]){self->kernel_num, output->shape[1], output->shape[2]});
        initialize_weights(self->weights);
        nda_init_rand(self->bias);
        nda_div_scalar(self->weights, 100, self->weights);
        printf("Weights shape : "); nda_print_shape(self->weights);
        printf("Bias shape : "); nda_print_shape(self->bias); printf("\n");

        self->weights_grad = nda_zero(4, (int[]){self->kernel_num, input->shape[0], self->kernel_size, self->kernel_size});
        self->bias_grad = nda_zero(3, (int[]){self->kernel_num, output->shape[1], output->shape[2]});
        self->linear_output = nda_zero(3, (int[]){self->kernel_num, output->shape[1], output->shape[2]});
    }
    self->input = input;
    nda_conv3d(input, self->weights, self->linear_output);
    nda_add(self->linear_output, self->bias, self->linear_output);
    activation_functions[self->activation](self->linear_output, output);
}

static void conv_backward(ConvLayer *self, ndarray *input_grad, ndarray *output_grad){
    // Calculate bias gradient
    activation_function_derivatives[self->activation](self->linear_output, self->linear_output);
    nda_mul(input_grad, self->linear_output, self->bias_grad);
    
    // Calculate weights gradient
    ndarray *X_c = nda_zero(2, (int[]){self->input->shape[1], self->input->shape[2]});
    ndarray *W_n = nda_zero(2, (int[]){self->bias_grad->shape[1], self->bias_grad->shape[2]});
    ndarray *K_nc = nda_zero(2, (int[]){self->weights->shape[2], self->weights->shape[3]});

    for (int n = 0; n < self->kernel_num; n++) {
        memcpy(W_n->data, self->bias_grad->data + n * W_n->size, W_n->size * sizeof(float));
        for (int c = 0; c < self->input->shape[0]; c++) {
            memcpy(X_c->data, self->input->data + c * X_c->size, X_c->size * sizeof(float));
            nda_conv2d(X_c, W_n, K_nc);
            memcpy(self->weights_grad->data + n * W_n->size + c * K_nc->size, K_nc->data, K_nc->size * sizeof(float));
        }
    }
    // If output_grad is not NULL, continue backpropagation
    if (output_grad != NULL) {
        ndarray *W_n_pad = nda_zero(2, (int[]){self->bias_grad->shape[1] + 2 * self->kernel_size - 2, 
                                           self->bias_grad->shape[2] + 2 * self->kernel_size - 2});
        ndarray *Output_c = nda_zero(2, (int[]){output_grad->shape[1], output_grad->shape[2]});

        for (int n = 0; n < self->kernel_num; n++) {
            memcpy(W_n->data, self->bias_grad->data + n * W_n->size, W_n->size * sizeof(float));
            nda_pad(W_n, self->kernel_size - 1, W_n_pad);
            for (int c = 0; c < self->input->shape[0]; c++) {
                memcpy(K_nc->data, self->weights->data + n * W_n->size + c * K_nc->size, K_nc->size * sizeof(float));
                nda_flip(K_nc);
                nda_conv2d(W_n_pad, K_nc, Output_c);
                for (int i = 0; i < output_grad->shape[1]; i++) {
                    for (int j = 0; j < output_grad->shape[2]; j++) {
                        output_grad->data[c * output_grad->strides[0] + i * output_grad->strides[1] + j] += 
                            Output_c->data[i * Output_c->strides[0] + j];
                    }
                }
                nda_flip(K_nc);
            }
        }
        nda_free(W_n_pad);
        nda_free(Output_c);
    }
    nda_free(X_c);
    nda_free(W_n);
    nda_free(K_nc);
}

ConvLayer *create_conv_layer(int kernel_num, int kernel_size, ActivationType activation){
    ConvLayer *layer = malloc(sizeof(ConvLayer));
    layer->activation = activation;
    layer->kernel_num = kernel_num;
    layer->kernel_size = kernel_size;
    layer->input = NULL;
    layer->weights = NULL;
    layer->bias = NULL;
    layer->weights_grad = NULL;
    layer->bias_grad = NULL;
    layer->linear_output = NULL;
    layer->forward = conv_forward;
    layer->backward = conv_backward;
    return layer;
}

void free_conv_layer(ConvLayer *layer){
    if (layer->weights != NULL) nda_free(layer->weights);
    if (layer->bias != NULL) nda_free(layer->bias);
    if (layer->weights_grad != NULL) nda_free(layer->weights_grad);
    if (layer->bias_grad != NULL) nda_free(layer->bias_grad);
    if (layer->linear_output != NULL) nda_free(layer->linear_output);
    free(layer);
}

static void flatten_forward(ndarray *input, ndarray *output){
    memcpy(output->data, input->data, input->size * sizeof(float));
}

static void flatten_backward(ndarray *input_grad, ndarray *output_grad){
    if (output_grad != NULL){
        memcpy(input_grad->data, output_grad->data, output_grad->size * sizeof(float));
    }
}

FlattenLayer *create_flatten_layer(){
    FlattenLayer *layer = malloc(sizeof(FlattenLayer));

    layer->forward = flatten_forward;
    layer->backward = flatten_backward;
    return layer;
}

void free_flatten_layer(FlattenLayer *layer){
    free(layer);
}

void save_dense_layer(DenseLayer *layer, FILE *file){
    // Write the shape of the weights and bias
    fprintf(file, "%d %d\n", layer->weights->shape[0], layer->weights->shape[1]);
    fprintf(file, "%d %d\n", layer->bias->shape[0], layer->bias->shape[1]);
    fprintf(file, "%d %d\n", layer->linear_output->shape[0], layer->linear_output->shape[1]);

    // Write weights
    for (int i = 0; i < layer->weights->size; i++) {
        fprintf(file, "%f ", layer->weights->data[i]);
    }

    fprintf(file, "\n");

    // Write bias
    for (int i = 0; i < layer->bias->size; i++) {
        fprintf(file, "%f ", layer->bias->data[i]);
    }

    // Add a separator
    fprintf(file, "\n---\n");
}

void save_conv_layer(ConvLayer *layer, FILE *file){
    // Write the shape of the weights and bias
    fprintf(file, "%d %d %d %d\n", layer->weights->shape[0], layer->weights->shape[1], layer->weights->shape[2], layer->weights->shape[3]);
    fprintf(file, "%d %d %d\n", layer->bias->shape[0], layer->bias->shape[1], layer->bias->shape[2]);

    // Write weights
    for (int i = 0; i < layer->weights->size; i++) {
        fprintf(file, "%f ", layer->weights->data[i]);
    }

    fprintf(file, "\n");

    // Write bias
    for (int i = 0; i < layer->bias->size; i++) {
        fprintf(file, "%f ", layer->bias->data[i]);
    }

    // Add a separator
    fprintf(file, "\n---\n");
}

void load_dense_layer(DenseLayer *layer, FILE *file) {
    char separator[5];
    // Read the shape of the weights and bias
    int weights_rows, weights_cols;
    int bias_rows, bias_cols;
    int linear_output_rows, linear_output_cols;
    fscanf(file, "%d %d", &weights_rows, &weights_cols);
    fscanf(file, "%d %d", &bias_rows, &bias_cols);
    fscanf(file, "%d %d", &linear_output_rows, &linear_output_cols);
    
    layer->weights = nda_zero(2, (int[]){weights_rows, weights_cols});
    layer->bias = nda_zero(2, (int[]){bias_rows, bias_cols});
    layer->linear_output = nda_zero(2, (int[]){linear_output_rows, linear_output_cols});

    // Read weights
    for (int i = 0; i < layer->weights->size; i++) {
        fscanf(file, "%f", &(layer->weights->data[i]));
    }
    // Read bias
    for (int i = 0; i < layer->bias->size; i++) {
        fscanf(file, "%f", &(layer->bias->data[i]));
    }
    // Read the separator
    fscanf(file, "%s", separator);
}

void load_conv_layer(ConvLayer *layer, FILE *file) {
    // Read the shape of the weights and bias
    int kernel_num, channels, kernel_size1, kernel_size2;
    int bias_channels, bias_rows, bias_cols;
    fscanf(file, "%d %d %d %d", &kernel_num, &channels, &kernel_size1, &kernel_size2);
    fscanf(file, "%d %d %d", &bias_channels, &bias_rows, &bias_cols);
    layer->kernel_num = kernel_num;
    layer->kernel_size = kernel_size1;
    layer->weights = nda_zero(4, (int[]){kernel_num, channels, kernel_size1, kernel_size2});
    layer->bias = nda_zero(3, (int[]){bias_channels, bias_rows, bias_cols});

    char separator[5];

    // Read weights
    for (int i = 0; i < layer->weights->size; i++) {
        fscanf(file, "%f", &(layer->weights->data[i]));
    }

    // Read bias
    for (int i = 0; i < layer->bias->size; i++) {
        fscanf(file, "%f", &(layer->bias->data[i]));
    }

    // Read the separator
    fscanf(file, "%s", separator);
}

void copy_dense_layer(DenseLayer *dst, DenseLayer *src){
    dst->weights = nda_deepcopy(src->weights);
    dst->bias = nda_deepcopy(src->bias);
    dst->linear_output = nda_zero(2, (int[]){src->linear_output->shape[0], src->linear_output->shape[1]});
}

void copy_conv_layer(ConvLayer *dst, ConvLayer *src){
    dst->kernel_num = src->kernel_num;
    dst->kernel_size = src->kernel_size;
    dst->weights = nda_deepcopy(src->weights);
    dst->bias = nda_deepcopy(src->bias);
    dst->linear_output = nda_zero(3, (int[]){src->linear_output->shape[0], src->linear_output->shape[1], src->linear_output->shape[2]});
}