#include "activation.h"
#include <math.h>

ActivationFunction *get_activation_function(ActivationFunctionType activation_type, float* parameters){
    ActivationFunction *activation_function = (ActivationFunction *)malloc(sizeof(ActivationFunction));
    activation_function->type = activation_type;
    activation_function->parameters = parameters;
    switch (activation_type){
        case SIGMOID:
            activation_function->function = matrix_sigmoid;
            activation_function->derivative = matrix_sigmoid_derivative;
            break;
        case TANH:
            activation_function->function = matrix_tanh;
            activation_function->derivative = matrix_tanh_derivative;
            break;
        case RELU:
            activation_function->function = matrix_relu;
            activation_function->derivative = matrix_relu_derivative;
            break;
        case LEAKY_RELU:
            activation_function->function = matrix_leaky_relu;
            activation_function->derivative = matrix_leaky_relu_derivative;
            break;
        case SOFTMAX:
            activation_function->function = matrix_softmax;
            activation_function->derivative = matrix_softmax_derivative;
            break;
        default:
            assert("Activation function not implemented yet.\n");
    }
}

void matrix_sigmoid(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        output->data[i] = sigmoid(input->data[i]);
    }
}

void matrix_sigmoid_derivative(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        output->data[i] = sigmoid_derivative(input->data[i]);
    }
}

void matrix_tanh(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        output->data[i] = tanh(input->data[i]);
    }
}

void matrix_tanh_derivative(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i = 0; i < input->rows * input->cols; ++i) {
        output->data[i] = tanh_derivative(input->data[i]);
    }
}

void matrix_relu(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i=0; i < input->rows * input->cols; ++i) {
        output->data[i] = relu(input->data[i]);
    }
}

void matrix_relu_derivative(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    for (size_t i=0; i < input->rows * input->cols; ++i) {
        output->data[i] = relu_derivative(input->data[i]);
    }
}

void matrix_leaky_relu(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    float alpha = activation_function->parameters[0];
    for (size_t i=0; i < input->rows * input->cols; ++i) {
        output->data[i] = leaky_relu(input->data[i], alpha);
    }
}

void matrix_leaky_relu_derivative(Matrix *output, const Matrix *input, ActivationFunction *activation_function) {
    assert_same_size(output, input);
    float alpha = activation_function->parameters[0];
    for (size_t i=0; i < input->rows * input->cols; ++i) {
        output->data[i] = leaky_relu_derivative(input->data[i], alpha);
    }
}

void matrix_softmax(Matrix *output, const Matrix *input) {
    assert_same_size(output, input);
    
    for (size_t i = 0; i < input->rows; ++i) {
        // Find the maximum element for each row
        float max_element = input->data[i * input->cols];
        for (size_t j = 1; j < input->cols; ++j) {
            if (input->data[i * input->cols + j] > max_element) {
                max_element = input->data[i * input->cols + j];
            }
        }

        // Compute the exponential of the input elements subtracting the maximum element
        float exp_sum = 0.0f;
        for (size_t j = 0; j < input->cols; ++j) {
            output->data[i * input->cols + j] = expf(input->data[i * input->cols + j] - max_element);
            exp_sum += output->data[i * input->cols + j];
        }

        // Normalize the output elements
        for (size_t j = 0; j < input->cols; ++j) {
            output->data[i * output->cols + j] /= exp_sum;
        }
    }
}

void matrix_softmax_derivative(Matrix *output, const Matrix *input, const Matrix *softmax_output) {
    assert_same_size(output, input);
    assert_same_size(output, softmax_output);
    
    for (size_t i = 0; i < input->rows; ++i) {
        for (size_t j = 0; j < input->cols; ++j) {
            float softmax_value = softmax_output->data[i * softmax_output->cols + j];
            output->data[i * output->cols + j] = softmax_value * (1.0f - softmax_value);
        }
    }
}

float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float sigmoid_derivative(float x) {
    float sigmoid_value = sigmoid(x);
    return sigmoid_value * (1.0f - sigmoid_value);
}

float tanh(float x) {
    return tanhf(x);
}

float tanh_derivative(float x) {
    float tanh_value = tanh(x);
    return 1.0f - tanh_value * tanh_value;
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

float relu_derivative(float x) {
    return x > 0.0f ? 1.0f : 0.0f;
}

float leaky_relu(float x, float alpha) {
    return x > 0.0f ? x : alpha * x;
}

float leaky_relu_derivative(float x, float alpha) {
    return x > 0.0f ? 1.0f : alpha;
}