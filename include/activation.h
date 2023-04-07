#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "matrix.h"

// Activation function enumeration
typedef enum {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SOFTMAX
    // Add more activation functions as needed
} ActivationFunctionType;

typedef void (*MatrixActivationFunction)(Matrix *output, const Matrix *input, ActivationFunction *activation_function);

typedef struct ActivationFunction {
    ActivationFunctionType type;
    float* parameters;
    MatrixActivationFunction function;
    MatrixActivationFunction derivative;
} ActivationFunction;

ActivationFunction *get_activation_function(ActivationFunctionType activation_type, float* parameters);
void free_activation_function(ActivationFunction *activation_function);

#endif // ACTIVATIONS_H
