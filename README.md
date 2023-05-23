# Neural Networks written in C

This work is related to the IN104 project at ENSTA Paris, specifically focusing on the implementation and testing of the neural network component.

## Usage

Modify the `network.h` and `network.c` file to change the network architecture. It relies on the `ndarray` and `layer` files.

Support layer types:

- Dense
- Conv
- Flatten

Support activation functions:

- Sigmoid
- ReLU
- Softmax

Support loss functions:

- MSE
- Cross Entropy

## Examples

The dataset used in the examples is the 20x20 MNIST dataset.

To train the network, run the following command:

```bash
cd ./examples
make
./mnist_train.x
```

The log file is saved in the `../logs` directory as `log_<timestamp>.txt`.

To test the network, run the following command:

```bash
./mnist_test.x <path_to_model>
```

The model is saved in the `../models` directory as `network_<timestamp>.txt`. The file `../models/network_network_2023_5_18_18_47_42.txt` is a trained model.

Then enter the index of the image in the test dataset you want to test, for example `0.341`.
