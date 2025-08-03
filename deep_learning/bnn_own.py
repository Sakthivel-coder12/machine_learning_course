import numpy as np

# Print all floats with exactly 4 digits after decimal point
np.set_printoptions(precision=4, suppress=True, floatmode='fixed')


def relu(z):
    return np.maximum(0, z)


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def get_input():
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
    
    print(f"\nEnter weights from input to hidden neuron:")
    W1 = np.zeros((input_size, 1))
    for i in range(input_size):
        weight = float(input(f"Weight from input {i+1}: "))
        W1[i, 0] = weight
    
    b1 = float(input("Enter hidden neuron bias: "))
    b1 = np.array([[b1]])
    
    W2 = float(input("\nEnter weight from hidden to output: "))
    W2 = np.array([[W2]])
    
    b2 = float(input("Enter output neuron bias: "))
    b2 = np.array([[b2]])
    
    epo = int(input("Enter how many epochs you want: "))
    
    # Training loop
    for epoch in range(epo):
        print(f"\n========== EPOCH {epoch+1} ==========")
        W1, b1, W2, b2 = forward_propagation(X, W1, b1, W2, b2, learning_rate, y_true)


def forward_propagation(X, W1, b1, W2, b2, learning_rate, y_true):
    print(f"\n=== FORWARD PROPAGATION ===")
    
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    
    print(f"Hidden (ReLU):")
    print(f"  z1 = {z1}")
    print(f"  a1 = max(0, z1) = {a1}")
    
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    
    print(f"\nOutput (Sigmoid):")
    print(f"  z2 = {z2}")
    print(f"  a2 = sigmoid(z2) = {a2}")
    
    print(f"\nPrediction: {a2}")
    print(f"True label: {y_true}")
    print(f"Error: {a2 - y_true}")
    
    return backward_propagation(X, W1, b1, W2, b2, learning_rate, y_true, a2, a1)


def backward_propagation(X, W1, b1, W2, b2, learning_rate, y_true, a2, a1):
    print(f"\n=== BACKWARD PROPAGATION ===")
    
    print(f"\nStep 1: Output layer error")
    dl_dY = -((y_true / a2) + ((1 - y_true) / (1 - a2)))
    dY_dz2 = a2 * (1 - a2)
    dz2_dw2 = a1
    dl_db2 = 1
    dl_dw2 = dl_dY * dY_dz2 * dz2_dw2
    dl_db2 = dl_dY * dY_dz2 * dl_db2
    w2new = W2 - learning_rate * dl_dw2
    b2new = b2 - learning_rate * dl_db2

    print(f"w2 = {W2} --> {w2new.T}")
    print(f"b2 = {b2} --> {b2new.T}")

    print(f"\nStep 2: Hidden layer error")
    
    dz2_da1 = W2
    da1_dz1 = a1 * (1-a1)
    dz1_dw1 = X
    dz1_b1 = 1
    
    dl_da1 = dl_dY * dY_dz2 * dz2_da1
    dl_dw1 = dl_da1 * da1_dz1 * dz1_dw1
    dl_db1 = dl_da1 * da1_dz1 * dz1_b1
    
    w1new = W1 - learning_rate * dl_dw1.T
    b1new = b1 - learning_rate * dl_db1

    print(f"w1 = {W1.T} --> {w1new.T}")
    print(f"b1 = {b1} --> {b1new}")
    
    return w1new, b1new, w2new, b2new


if __name__ == "__main__":
    get_input()
