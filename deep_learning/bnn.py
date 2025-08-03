import numpy as np

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def sigmoid(z):
    """Sigmoid activation function"""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def backward_propagation():
    """
    Backward Propagation: One hidden neuron (ReLU) + One output neuron (Sigmoid)
    """
    print("=== BACKWARD PROPAGATION - BINARY CLASSIFICATION ===")
    print("Network: Input → [1 Hidden Neuron (ReLU)] → [1 Output Neuron (Sigmoid)]")
    print()
    
    # Get inputs
    input_size = int(input("Enter number of input features: "))
    
    print(f"\nEnter {input_size} input values:")
    X = []
    for i in range(input_size):
        value = float(input(f"Input x{i+1}: "))
        X.append(value)
    X = np.array(X).reshape(1, -1)
    
    y_true = float(input(f"\nEnter true label (0 or 1): "))
    y_true = np.array([[y_true]])
    
    learning_rate = float(input("Enter learning rate: "))
    
    # Get current weights
    print(f"\n=== CURRENT WEIGHTS ===")
    
    # Hidden layer weights (ReLU)
    print(f"\nEnter weights from input to hidden neuron:")
    W1 = np.zeros((input_size, 1))
    for i in range(input_size):
        weight = float(input(f"Weight from input {i+1}: "))
        W1[i, 0] = weight
    
    b1 = float(input("Enter hidden neuron bias: "))
    b1 = np.array([[b1]])
    
    # Output layer weight (Sigmoid)
    W2 = float(input("\nEnter weight from hidden to output: "))
    W2 = np.array([[W2]])
    
    b2 = float(input("Enter output neuron bias: "))
    b2 = np.array([[b2]])
    
    # Forward propagation
    print(f"\n=== FORWARD PROPAGATION ===")
    
    # Hidden layer (ReLU)
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    
    print(f"Hidden (ReLU):")
    print(f"  z1 = {z1.flatten()[0]:.6f}")
    print(f"  a1 = max(0, z1) = {a1.flatten()[0]:.6f}")
    
    # Output layer (Sigmoid)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    print(f"\nOutput (Sigmoid):")
    print(f"  z2 = {z2.flatten()[0]:.6f}")
    print(f"  a2 = sigmoid(z2) = {a2.flatten()[0]:.6f}")
    
    print(f"\nPrediction: {a2.flatten()[0]:.6f}")
    print(f"True label: {y_true.flatten()[0]}")
    print(f"Error: {(a2.flatten()[0] - y_true.flatten()[0]):.6f}")
    
    # Backward propagation
    print(f"\n=== BACKWARD PROPAGATION ===")
    
    # Step 1: Output layer error (sigmoid)
    print(f"\nStep 1: Output layer error")
    delta_output = (a2 - y_true) * a2 * (1 - a2)
    print(f"δ_output = (ŷ - y) × sigmoid(z2) × (1 - sigmoid(z2)) = {delta_output.flatten()[0]:.6f}")
    
    # Step 2: Hidden layer error (ReLU)
    print(f"\nStep 2: Hidden layer error")
    relu_derivative_value = 1 if z1.flatten()[0] > 0 else 0
    delta_hidden = delta_output * W2 * relu_derivative_value
    print(f"δ_hidden = δ_output × W2 × relu'(z1)")
    print(f"relu'(z1) = {relu_derivative_value} (since z1 = {z1.flatten()[0]:.6f})")
    print(f"δ_hidden = {delta_hidden.flatten()[0]:.6f}")
    
    # Step 3: Calculate gradients
    print(f"\nStep 3: Calculate gradients")
    
    dW2 = a1 * delta_output
    db2 = delta_output
    dW1 = np.dot(X.T, delta_hidden)
    db1 = delta_hidden
    
    print(f"dW2 = {dW2.flatten()[0]:.6f}")
    print(f"db2 = {db2.flatten()[0]:.6f}")
    print(f"dW1 = {dW1.flatten()}")
    print(f"db1 = {db1.flatten()[0]:.6f}")
    
    # Step 4: Update weights
    print(f"\nStep 4: Update weights")
    
    W2_new = W2 - learning_rate * dW2
    b2_new = b2 - learning_rate * db2
    W1_new = W1 - learning_rate * dW1
    b1_new = b1 - learning_rate * db1
    
    print(f"\nWeight updates:")
    print(f"W2: {W2.flatten()[0]:.6f} → {W2_new.flatten()[0]:.6f}")
    print(f"b2: {b2.flatten()[0]:.6f} → {b2_new.flatten()[0]:.6f}")
    print(f"W1: {W1.flatten()} → {W1_new.flatten()}")
    print(f"b1: {b1.flatten()[0]:.6f} → {b1_new.flatten()[0]:.6f}")
    
    # Test new weights
    print(f"\n=== FORWARD PASS WITH NEW WEIGHTS ===")
    
    z1_new = np.dot(X, W1_new) + b1_new
    a1_new = relu(z1_new)
    z2_new = np.dot(a1_new, W2_new) + b2_new
    a2_new = sigmoid(z2_new)
    
    print(f"New prediction: {a2_new.flatten()[0]:.6f}")
    print(f"Old prediction: {a2.flatten()[0]:.6f}")
    print(f"True label: {y_true.flatten()[0]}")
    
    old_error = abs(a2.flatten()[0] - y_true.flatten()[0])
    new_error = abs(a2_new.flatten()[0] - y_true.flatten()[0])
    print(f"Error: {old_error:.6f} → {new_error:.6f}")
    print(f"Improvement: {old_error - new_error:.6f}")

if __name__ == "__main__":
    backward_propagation()