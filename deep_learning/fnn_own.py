import numpy as np

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

# Input layer
n = int(input("Enter number of input features: "))
x_vals = list(map(float, input(f"Enter {n} input values: ").split()))
x = np.array(x_vals).reshape(1, -1)

a = x
layers = int(input("Enter number of hidden layers: "))
layer_dims = []

# Hidden layers
for l in range(layers):
    print(f"\n=== Hidden Layer {l+1} ===")
    input_dim = a.shape[1]
    output_dim = int(input(f"Enter number of neurons in layer {l+1}: "))
    layer_dims.append((input_dim, output_dim))

    print(f"Enter {input_dim * output_dim} weights (row-wise) separated by space:")
    weight_values = list(map(float, input().split()))
    W = np.array(weight_values).reshape(input_dim, output_dim)

    print(f"Enter {output_dim} biases separated by space:")
    bias_values = list(map(float, input().split()))
    b = np.array(bias_values).reshape(1, output_dim)

    z = np.dot(a, W) + b
    print(f"Z = {z}")
    a = relu(z)
    print(f"A (after ReLU) = {a}")

# Output layer
print(f"\n=== Output Layer ===")
output_dim = int(input("Enter number of output neurons: "))
input_dim = a.shape[1]

print(f"Enter {input_dim * output_dim} weights (row-wise) separated by space:")
output_weights = list(map(float, input().split()))
W_out = np.array(output_weights).reshape(input_dim, output_dim)

print(f"Enter {output_dim} biases separated by space:")
output_biases = list(map(float, input().split()))
b_out = np.array(output_biases).reshape(1, output_dim)

z_out = np.dot(a, W_out) + b_out
a_out = sigmoid(z_out)

print(f"\nOutput Z = {z_out}")
print(f"Output A (after Sigmoid) = {a_out}")