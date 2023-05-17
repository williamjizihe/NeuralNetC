#include "layer.h"

#include <string.h>
#include <stdlib.h>

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
        self->weights = nda_zero(2, (size_t[]){output->shape[0], input->shape[0]});
        self->bias = nda_zero(2, (size_t[]){output->shape[0], 1});
        initialize_weights(self->weights);
        for (size_t i = 0; i < self->bias->size; i++) {
            self->bias->data[i] = 0.1;
        }

        nda_div_scalar(self->weights, 100, self->weights);
        nda_div_scalar(self->bias, 100, self->bias);
        self->weights_grad = nda_zero(2, (size_t[]){output->shape[0], input->shape[0]});
        self->bias_grad = nda_zero(2, (size_t[]){output->shape[0], 1});
        self->linear_output = nda_zero(2, (size_t[]){output->shape[0], 1});
    }
    self->input = input;
    nda_dot(self->weights, input, self->linear_output);
    nda_add(output, self->bias, self->linear_output);
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

void free_conv_layer(ConvLayer *layer){
    if (layer->weights != NULL) nda_free(layer->weights);
    if (layer->bias != NULL) nda_free(layer->bias);
    if (layer->weights_grad != NULL) nda_free(layer->weights_grad);
    if (layer->bias_grad != NULL) nda_free(layer->bias_grad);
    free(layer);
}