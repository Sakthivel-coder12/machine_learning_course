import numpy as np

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    z = np.clip(z, -500, 500)  
    return 1 / (1 + np.exp(-z))

def forward_propagation():
    print("=== FORWARD PROPAGATION - BINARY CLASSIFICATION ===\n")
    
    # Get number of input features
    input_size = int(input("Enter number of input features: "))
    
    # Get input values
    print(f"\nEnter {input_size} input values:")
    X = []
    for i in range(input_size):
        value = float(input(f"Input x{i+1}: "))
        X.append(value)
    X = np.array(X).reshape(1, -1)
    
    print(f"\nInput vector: {X.flatten()}")
    
    # Get number of hidden layers
    num_hidden_layers = int(input("\nEnter number of hidden layers: "))
    
    current_input = X
    
    # Process each hidden layer (using ReLU)
    for layer in range(num_hidden_layers):
        print(f"\n--- Hidden Layer {layer + 1} (ReLU) ---")
        
        num_neurons = int(input(f"Enter number of neurons in hidden layer {layer + 1}: "))
        
        # Get weights and biases
        input_dim = current_input.shape[1]
        weights = np.zeros((input_dim, num_neurons))
        biases = np.zeros((1, num_neurons))
        
        print(f"\nEnter weights ({input_dim} x {num_neurons}):")
        for i in range(input_dim):
            for j in range(num_neurons):
                weight = float(input(f"Weight from input {i+1} to neuron {j+1}: "))
                weights[i, j] = weight
        
        print(f"\nEnter biases:")
        for j in range(num_neurons):
            bias = float(input(f"Bias for neuron {j+1}: "))
            biases[0, j] = bias
        
        # Forward pass with ReLU
        z = np.dot(current_input, weights) + biases
        a = relu(z)
        
        print(f"\nLayer {layer + 1} Results:")
        print(f"Z (before ReLU): {z.flatten()}")
        print(f"A (after ReLU): {a.flatten()}")
        
        current_input = a
    

    print(f"\n--- Output Layer (Sigmoid) ---")
    
    input_dim = current_input.shape[1]
    print(f"\nEnter weights for output layer ({input_dim} x 1):")
    output_weights = np.zeros((input_dim, 1))
    for i in range(input_dim):
        weight = float(input(f"Weight from hidden neuron {i+1} to output: "))
        output_weights[i, 0] = weight
    
    output_bias = float(input("Enter bias for output neuron: "))
    output_bias = np.array([[output_bias]])
    
    # Forward pass with sigmoid
    z_output = np.dot(current_input, output_weights) + output_bias
    a_output = sigmoid(z_output)
    
    print(f"\nOutput Layer Results:")
    print(f"Z (before sigmoid): {z_output.flatten()[0]:.6f}")
    print(f"A (after sigmoid): {a_output.flatten()[0]:.6f}")
    
    # Final prediction
    prediction = 1 if a_output.flatten()[0] > 0.5 else 0
    probability = a_output.flatten()[0]
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Probability of Class 1: {probability:.6f}")
    print(f"Prediction: Class {prediction}")

if __name__ == "__main__":
    forward_propagation()