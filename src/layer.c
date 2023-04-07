#include "layer.h"

LayerSpec *create_dense_layer(LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_function_type){
    LayerSpec *layer = (LayerSpec *)malloc(sizeof(LayerSpec));
    layer->type = layer_type;
    layer->epoch = 0;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_function = create_activation_function(activation_function_type);

    layer->optimizer = NULL;
    layer->next = NULL;
    layer->prev = NULL;    
    layer->accumulated_gradient = NULL;
    layer->accumulated_moment = NULL;
    layer->size_t_params = NULL;

    layer->weights = create_matrix(input_size.cols*input_size.rows, output_size.cols*output_size.rows);
    layer->biases = create_matrix(1, output_size.cols*output_size.rows);
    layer->output = create_matrix(output_size.rows, output_size.cols);
    layer->weight_gradient = create_matrix(input_size.cols*input_size.rows, output_size.cols*output_size.rows);
    layer->bias_gradient = create_matrix(1, output_size.cols*output_size.rows);
    layer->forward = dense_forward;
    layer->backward = dense_backward;

    return layer;
}

LayerSpec *create_convolution_layer(LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_function_type, size_t kernel_rows, size_t kernel_cols, size_t padding, size_t stride){
    LayerSpec *layer = (LayerSpec *)malloc(sizeof(LayerSpec));
    layer->type = layer_type;
    layer->epoch = 0;
    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation_function = create_activation_function(activation_function_type);

    layer->optimizer = NULL;
    layer->next = NULL;
    layer->prev = NULL;    
    layer->accumulated_gradient = NULL;
    layer->accumulated_moment = NULL;
    layer->size_t_params = (size_t *)malloc(4*sizeof(size_t));
    layer->size_t_params[0] = kernel_rows;
    layer->size_t_params[1] = kernel_cols;
    layer->size_t_params[2] = padding;
    layer->size_t_params[3] = stride;

    layer->weights = create_matrix(kernel_rows * kernel_cols, output_size.cols * output_size.rows);
    layer->biases = create_matrix(1, output_size.cols * output_size.rows);
    layer->output = create_matrix(output_size.rows, output_size.cols);
    layer->weight_gradient = create_matrix(kernel_rows * kernel_cols, output_size.cols * output_size.rows);
    layer->bias_gradient = create_matrix(1, output_size.cols * output_size.rows);

    layer->forward = conv_forward;
    layer->backward = conv_backward;

    return layer;
}

void free_layer(LayerSpec *layer){
    free_activation_function(layer->activation_function);
    free_matrix(layer->weights);
    free_matrix(layer->biases);
    free_matrix(layer->output);
    free_matrix(layer->weight_gradient);
    free_matrix(layer->bias_gradient);
    if (layer->accumulated_gradient != NULL){
        free_matrix(layer->accumulated_gradient);
    }
    if (layer->accumulated_moment != NULL){
        free_matrix(layer->accumulated_moment);
    }
    if (layer->size_t_params != NULL){
        free(layer->size_t_params);
    }
    free(layer);
}

void dense_forward(LayerSpec *current, const Matrix *input) {
    // Z = input * weights + biases
    matrix_multiply(current->output, input, current->weights);
    matrix_add(current->output, current->output, current->biases);
    
    // A = activation_function(Z)
    current->activation_function->function(current->output, current->output, current->activation_function);
    current->output->rows = current->output_size.rows;
    current->output->cols = current->output_size.cols;
}

void dense_backward(LayerSpec *current, const Matrix *output_gradient, Matrix *input_gradient) {
    // Compute activation function's derivative: dA/dZ
    Matrix *dAdZ = create_matrix(current->output->rows, current->output->cols);
    current->activation_function->derivative(dAdZ, current->output, current->activation_function);

    // Compute dL/dZ = dL/dA * dA/dZ (element-wise multiplication)
    matrix_elementwise_multiply(dAdZ, output_gradient, dAdZ);

    // Compute dL/dW = input^T * dL/dZ
    Matrix *input_transpose = create_matrix(current->prev->output->cols, current->prev->output->rows);
    matrix_transpose(input_transpose, current->prev->output);
    matrix_multiply(current->weight_gradient, input_transpose, dAdZ);

    // Compute dL/dB = sum(dL/dZ, axis=0)
    matrix_sum_axis(current->bias_gradient, dAdZ, 0);

    // Compute dL/dX = dL/dZ * weights^T
    Matrix *weights_transpose = create_matrix(current->weights->cols, current->weights->rows);
    matrix_transpose(weights_transpose, current->weights);
    matrix_multiply(input_gradient, dAdZ, weights_transpose);

    // Free temporary matrices
    free_matrix(dAdZ);
    free_matrix(input_transpose);
    free_matrix(weights_transpose);
}

void conv_forward(LayerSpec *current, const Matrix *input) {
    /*
        Z = conv(input, weights) + biases
        A = activation_function(Z)
    */
    size_t padding = current->size_t_params[2];
    size_t stride = current->size_t_params[3];
    matrix_convolve(current->output, input, current->weights, padding, stride);
    matrix_add(current->output, current->output, current->biases);
    current->activation_function->function(current->output, current->output, current->activation_function);
}

void conv_backward(LayerSpec *current, const Matrix *output_gradient, Matrix *input_gradient) {
    // TODO: Implement convolutional layer backward propagation
}