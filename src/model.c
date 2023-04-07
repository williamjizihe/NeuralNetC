#include <stdlib.h>
#include "model.h"

Model *create_model() {
    Model *model = (Model *)malloc(sizeof(Model));
    model->layers = NULL;
    model->num_layers = 0;
    model->loss_function = NULL;
    model->optimizer = NULL;
    return model;
}

void free_model(Model *model) {
    for (size_t i = 0; i < model->num_layers; i++) {
        free_layer(model->layers[i]);
    }
    free_optimizer(model->optimizer);
    free_loss_function(model->loss_function);
    free(model->layers);
    free(model);
}

void add_layer(Model *model, LayerType layer_type, MatrixSize input_size, MatrixSize output_size, ActivationFunctionType activation_type) {
    model->layers = (LayerSpec **)realloc(model->layers, (model->num_layers + 1) * sizeof(LayerSpec *));
    model->layers[model->num_layers] = create_layer(layer_type, input_size, output_size, activation_type);
    model->num_layers++;
}

void compile_model(Model *model, OptimizerType optimizer_type, float* optimizer_params, LossFunctionType loss_function_type){
    model->optimizer = create_optimizer(optimizer_type, optimizer_params);
    model->loss_function = create_loss_function(loss_function_type);

    for (size_t i = 0; i < model->num_layers; i++) {
        model->layers[i]->optimizer = model->optimizer;
        if (i > 0) {
            model->layers[i]->prev = model->layers[i - 1];
            model->layers[i - 1]->next = model->layers[i];
        }
        if (optimizer_type == OPTIMIZER_ADAM) {
            model->layers[i]->accumulated_gradient = create_matrix(model->layers[i]->weights->rows, model->layers[i]->weights->cols);
            model->layers[i]->accumulated_moment = create_matrix(model->layers[i]->weights->rows, model->layers[i]->weights->cols);
        }   
    }
}

void fit(Model *model, Matrix *X_train, Matrix *y_train, size_t epochs) {
    size_t num_samples = X_train->rows;

    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;

        for (size_t sample = 0; sample < num_samples; ++sample) {
            Matrix *input = get_row(X_train, sample);
            Matrix *label = get_row(y_train, sample);

            // Forward pass
            Matrix *prediction = forward_pass(model, input);

            // Calculate loss
            float loss = model->loss_function->loss(prediction, label);
            total_loss += loss;

            // Calculate gradients using backward pass
            Matrix *output_gradient = model->loss_function->gradient(prediction, label);
            backward_pass(model, output_gradient);

            // Update weights and biases using the optimizer
            update_weights_and_biases(model);

            // Free memory
            free_matrix(input);
            free_matrix(label);
            free_matrix(prediction);
            free_matrix(output_gradient);
        }

        // Calculate average loss
        float average_loss = total_loss / num_samples;
        printf("Epoch: %zu, Average Loss: %f\n", epoch + 1, average_loss);
    }
}
