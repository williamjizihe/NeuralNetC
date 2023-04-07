#ifndef MODEL_H
#define MODEL_H

#include "layer.h"
#include "loss_function.h"
#include "activation.h"
#include "optimizer.h"

typedef struct Model {
    LayerSpec **layers;
    size_t num_layers;
    LossFunctionSpec *loss_function;
    OptimizerSpec *optimizer;
} Model;

Model *create_model();
void free_model(Model *model);

void add_layer(Model *model, LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_type);

void compile_model(Model *model, OptimizerType optimizer_type, float* optimizer_params, LossFunctionType loss_function_type);
void fit(Model *model, Matrix *X_train, Matrix *y_train, size_t epochs);

#endif // MODEL_H
