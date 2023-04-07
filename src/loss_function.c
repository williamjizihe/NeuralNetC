#include "loss_function.h"

LossFunctionSpec *get_loss_function(LossFunctionType type) {
    LossFunctionSpec *spec = (LossFunctionSpec *)malloc(sizeof(LossFunctionSpec));
    spec->type = type;
    switch (type) {
        case LOSS_MSE:
            spec->function = mse_loss;
            spec->gradient = mse_loss_gradient;
            break;
        case LOSS_CROSS_ENTROPY:
            spec->function = cross_entropy_loss;
            spec->gradient = cross_entropy_loss_gradient;
            break;
        // Add more loss functions here
        default:
            break;
    }
    return spec;
}

void free_loss_function(LossFunctionSpec *loss_function){
    free(loss_function);
}

float mse_loss(Matrix *y_true, Matrix *y_pred) {
    assert_same_size(y_true, y_pred);
    float loss = 0.0;
    for (size_t i = 0; i < y_true->rows; i++) {
        for (size_t j = 0; j < y_true->cols; j++) {
            float diff = y_true->data[i][j] - y_pred->data[i][j];
            loss += diff * diff;
        }
    }
    return loss / (y_true->rows * y_true->cols);
}

void mse_loss_gradient(Matrix *result, Matrix *y_true, Matrix *y_pred) {
    assert_same_size(y_true, y_pred);
    for (size_t i = 0; i < y_true->rows; i++) {
        for (size_t j = 0; j < y_true->cols; j++) {
            result->data[i][j] = 2.0 * (y_pred->data[i][j] - y_true->data[i][j]);
        }
    }
}

float cross_entropy_loss(Matrix *y_true, Matrix *y_pred) {
    assert_same_size(y_true, y_pred);

    // Normalize y_pred matrix
    matrix_normalize(y_pred);

    float loss = 0.0;
    for (size_t i = 0; i < y_true->rows; i++) {
        for (size_t j = 0; j < y_true->cols; j++) {
            loss -= y_true->data[i][j] * log(y_pred->data[i][j]);
        }
    }
    return loss / y_true->rows;
}

void cross_entropy_loss_gradient(Matrix *result, Matrix *y_true, Matrix *y_pred) {
    assert_same_size(y_true, y_pred);

    // Normalize y_pred matrix
    matrix_normalize(y_pred);
    
    for (size_t i = 0; i < y_true->rows; i++) {
        for (size_t j = 0; j < y_true->cols; j++) {
            result->data[i][j] = -y_true->data[i][j] / y_pred->data[i][j];
        }
    }
}