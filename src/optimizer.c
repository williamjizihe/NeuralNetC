#include "optimizer.h"
#include "matrix.h"
#include <stdlib.h>
#include <math.h>

OptimizerSpec *create_optimizer(OptimizerType type, float *parameters) {
    OptimizerSpec *optimizer = (OptimizerSpec *) malloc(sizeof(OptimizerSpec));
    optimizer->learning_rate = parameters[0];
    optimizer->parameters = parameters;
    switch (type) {
        case OPTIMIZER_SGD: // Parameters: None
            optimizer->update_weights = sgd_update_weights;
            break;
        case OPTIMIZER_MOMENTUM: // Parameters: float momentum
            optimizer->update_weights = momentum_update_weights;
            break;
        case OPTIMIZER_RMSPROP: // Parameters: float decay_rate
            optimizer->update_weights = rmsprop_update_weights;
            break;
        case OPTIMIZER_ADAM: // Parameters: float beta1, float beta2, int t
            optimizer->update_weights = adam_update_weights;
            break;
    }
    return optimizer;
}

void free_optimizer(OptimizerSpec *optimizer){
    free(optimizer);
}

// Standard Gradient Descent (SGD) update function
void sgd_update_weights(LayerSpec *layer, Matrix *gradient, OptimizerSpec *optimizer) {
    Matrix *weights = layer->weights;
    float learning_rate = optimizer->learning_rate;

    matrix_scalar_multiply(gradient, gradient, -learning_rate);
    matrix_add(weights, weights, gradient);
}

// Momentum update function
void momentum_update_weights(LayerSpec *layer, Matrix *gradient, OptimizerSpec *optimizer) {
    Matrix *weights = layer->weights;
    Matrix *accumulated_moment = layer->accumulated_moment;
    float learning_rate = optimizer->learning_rate;
    float momentum = optimizer->parameters[1];

    matrix_scalar_multiply(accumulated_moment, accumulated_moment, momentum);
    matrix_scalar_multiply(gradient, gradient, -learning_rate);
    matrix_add(accumulated_moment, accumulated_moment, gradient);
    matrix_add(weights, weights, accumulated_moment);
}

// RMSprop update function
void rmsprop_update_weights(LayerSpec *layer, Matrix *gradient, OptimizerSpec *optimizer) {
    Matrix *weights = layer->weights;
    Matrix *accumulated_gradient = layer->accumulated_gradient;
    float learning_rate = optimizer->learning_rate;
    float decay = optimizer->parameters[1];

    matrix_elementwise_multiply(accumulated_gradient, accumulated_gradient, gradient);
    matrix_scalar_multiply(accumulated_gradient, accumulated_gradient, decay);
    matrix_scalar_multiply(gradient, gradient, 1.0 - decay);
    matrix_add(accumulated_gradient, accumulated_gradient, gradient);
    matrix_elementwise_divide_with_epsilon(gradient, gradient, accumulated_gradient, 1e-8);
    matrix_scalar_multiply(gradient, gradient, -learning_rate);
    matrix_add(weights, weights, gradient);
}

// Adam update function
void adam_update_weights(LayerSpec *layer, Matrix *gradient, OptimizerSpec *optimizer) {
    Matrix *weights = layer->weights;
    Matrix *accumulated_moment = layer->accumulated_moment;
    Matrix *accumulated_gradient = layer->accumulated_gradient;
    float learning_rate = optimizer->learning_rate;
    float beta1 = optimizer->parameters[1];
    float beta2 = optimizer->parameters[2];
    size_t t = layer->epoch;

    matrix_scalar_multiply(accumulated_moment, accumulated_moment, beta1);
    matrix_scalar_multiply(gradient, gradient, 1.0 - beta1);
    matrix_add(accumulated_moment, accumulated_moment, gradient);

    matrix_elementwise_multiply(accumulated_gradient, accumulated_gradient, gradient);
    matrix_scalar_multiply(accumulated_gradient, accumulated_gradient, beta2);
    matrix_scalar_multiply(gradient, gradient, 1.0 - beta2);
    matrix_add(accumulated_gradient, accumulated_gradient, gradient);

    Matrix *m_hat = create_matrix(weights->rows, weights->cols);
    matrix_scalar_multiply(m_hat, accumulated_moment, 1.0 / (1.0 - pow(beta1, t)));

    Matrix *v_hat = create_matrix(weights->rows, weights->cols);
    matrix_scalar_multiply(v_hat, accumulated_gradient, 1.0 / (1.0 - pow(beta2, t)));

    matrix_elementwise_divide_with_epsilon(gradient, m_hat, v_hat, 1e-8);
    matrix_scalar_multiply(gradient, gradient, -learning_rate);
    matrix_add(weights, weights, gradient);

    free_matrix(m_hat);
    free_matrix(v_hat);
}
