# MNIST Handwritten Digit Classifier

A neural network built from scratch using NumPy to classify handwritten digits (0-9) from the MNIST dataset.

## Features

- 2-layer neural network (784 → 10 → 10)
- ReLU activation for hidden layer
- Softmax activation for output layer
- Gradient descent optimization
- ~85% accuracy on training data

## Dataset

Uses MNIST CSV format from Kaggle:
- 60,000 training images
- 10,000 test images
- 28×28 pixel grayscale images

## Architecture

```
Input Layer (784)  →  Hidden Layer (10)  →  Output Layer (10)
                   ReLU                  Softmax
```

## Usage

```python
# Load and preprocess data
X_train = train_df.iloc[:, 1:].values.T / 255.0
y_train = train_df.iloc[:, 0].values

# Train the model
W1, B1, W2, B2 = gradient_descent(X_train, y_train, alpha=0.1, iterations=500)

# Make predictions
Z1, A1, Z2, A2 = forward_propagation(W1, B1, W2, B2, X_test)
predictions = get_predictions(A2)
```

## Key Functions

- `initialize_parameters()` - Random weight initialization
- `forward_propagation()` - Compute predictions
- `backward_propagation()` - Calculate gradients
- `update_parameters()` - Apply gradient descent

## Requirements

```
numpy
pandas
matplotlib
```

## Results

Achieves ~86% accuracy after 1000 iterations.

## License

MIT
