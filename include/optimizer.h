#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"
#include "layer.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM
} OptimizerType;

typedef void (*OptimizerFunction)(LayerSpec *layer, Matrix *gradient, OptimizerSpec *optimizer);

typedef struct Optimizer {
    float learning_rate;
    float* parameters;
    OptimizerFunction update_weights;
} OptimizerSpec;

OptimizerSpec *create_optimizer(OptimizerType type, float *parameters);
void free_optimizer(OptimizerSpec *optimizer);

#endif // OPTIMIZER_H
