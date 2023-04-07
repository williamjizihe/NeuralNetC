#ifndef LOSS_FUNCTION_H
#define LOSS_FUNCTION_H

#include "matrix.h"

typedef enum {
    LOSS_MSE,           // Mean Squared Error
    LOSS_CROSS_ENTROPY, // Cross-Entropy Loss
    // Add more loss functions here
} LossFunctionType;

typedef float (*LossFunction)(Matrix *y_true, Matrix *y_pred);
typedef void (*LossGradient)(Matrix *result, Matrix *y_true, Matrix *y_pred);

typedef struct LossFunction {
    LossFunctionType type;
    LossFunction function;
    LossGradient gradient;
} LossFunctionSpec;

LossFunctionSpec *get_loss_function(LossFunctionType type);
void free_loss_function(LossFunctionSpec *loss_function);

#endif // LOSS_FUNCTION_H
