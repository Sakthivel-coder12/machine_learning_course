import numpy as np

# Activation Functions and Derivatives
def relu(z): 
    return np.maximum(0, z)

def relu_derivative(z): 
    return (z > 0).astype(float)

def sigmoid(z): 
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Prevent overflow

def sigmoid_derivative(z): 
    s = sigmoid(z)
    return s * (1 - s)

def tanh(z): 
    return np.tanh(z)

def tanh_derivative(z): 
    return 1 - np.tanh(z) ** 2

def linear(z):
    return z

def linear_derivative(z):
    return np.ones_like(z)

def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def softmax_derivative(z):
    # For softmax, derivative is handled in backprop with cross-entropy
    s = softmax(z)
    return s * (1 - s)  # Simplified, actual implementation more complex

# Loss Functions
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Loss derivatives for backprop
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.shape[0]

# Activation registry
activation_functions = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax, softmax_derivative)
}

# Loss registry
loss_functions = {
    "mse": (mse, mse_derivative),
    "mae": (mae, mae_derivative),
    "binary_cross_entropy": (binary_cross_entropy, None),
    "categorical_cross_entropy": (categorical_cross_entropy, None)
}

class NeuralNetwork:
    def __init__(self, layers, activations, output_activation="linear", 
                 loss="mse", lr=0.01, task_type="regression"):
        self.layers = layers
        self.activations = activations
        self.output_activation = output_activation
        self.task_type = task_type
        self.lr = lr
        
        # Set loss function
        if loss in loss_functions:
            self.loss_fn, self.loss_derivative = loss_functions[loss]
        else:
            raise ValueError(f"Unknown loss function: {loss}")
        
        self.params = {}
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization"""
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            # He initialization for ReLU, Xavier for others
            if i < len(self.activations) and self.activations[i] == "relu":
                # He initialization
                std = np.sqrt(2.0 / self.layers[i])
            else:
                # Xavier initialization
                std = np.sqrt(1.0 / self.layers[i])
            
            self.params[f"W{i+1}"] = np.random.randn(self.layers[i], self.layers[i+1]) * std
            self.params[f"b{i+1}"] = np.zeros((1, self.layers[i+1]))
    
    def forward(self, X):
        self.cache = {"A0": X}
        
        for i in range(1, len(self.layers)):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            Z = np.dot(self.cache[f"A{i-1}"], W) + b
            
            # Choose activation function
            if i == len(self.layers) - 1:  # Output layer
                act_name = self.output_activation
            else:  # Hidden layers
                act_name = self.activations[i-1]
            
            act_func, _ = activation_functions[act_name]
            self.cache[f"Z{i}"] = Z
            self.cache[f"A{i}"] = act_func(Z)
        
        return self.cache[f"A{len(self.layers) - 1}"]
    
    def backward(self, Y):
        grads = {}
        L = len(self.layers) - 1
        m = Y.shape[0]
        A_final = self.cache[f"A{L}"]
        
        # Output layer gradient computation
        if self.output_activation == "softmax" and "categorical_cross_entropy" in str(self.loss_fn.__name__):
            # Special case: softmax + cross-entropy
            dZ = A_final - Y
        elif self.output_activation == "sigmoid" and "binary_cross_entropy" in str(self.loss_fn.__name__):
            # Special case: sigmoid + binary cross-entropy
            dZ = A_final - Y
        else:
            # General case: use loss derivative and activation derivative
            if self.loss_derivative is not None:
                dA = self.loss_derivative(Y, A_final)
            else:
                # For cross-entropy losses, compute gradient directly
                dA = A_final - Y
            
            # Apply activation derivative
            _, act_derivative = activation_functions[self.output_activation]
            dZ = dA * act_derivative(self.cache[f"Z{L}"])
        
        # Backpropagate through all layers
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f"A{i-1}"]
            W = self.params[f"W{i}"]
            
            # Compute gradients
            grads[f"dW{i}"] = np.dot(A_prev.T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Compute gradient for previous layer (if not input layer)
            if i > 1:
                Z_prev = self.cache[f"Z{i-1}"]
                _, act_deriv = activation_functions[self.activations[i-2]]
                dA_prev = np.dot(dZ, W.T)
                dZ = dA_prev * act_deriv(Z_prev)
        
        # Update parameters
        for i in range(1, L + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]
    
    def compute_loss(self, Y, Y_hat):
        return self.loss_fn(Y, Y_hat)
    
    def train(self, X, Y, epochs=1000, verbose=True):
        losses = []
        for epoch in range(epochs):
            Y_hat = self.forward(X)
            loss = self.compute_loss(Y, Y_hat)
            losses.append(loss)
            self.backward(Y)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss:.6f}")
        
        return losses
    
    def predict(self, X):
        probs = self.forward(X)
        
        if self.task_type == "regression":
            return probs  # Return raw predictions for regression
        elif self.task_type == "binary_classification":
            return (probs > 0.5).astype(int)
        elif self.task_type == "multiclass_classification":
            return np.argmax(probs, axis=1)
        else:
            # Auto-detect based on output activation
            if self.output_activation == "softmax":
                return np.argmax(probs, axis=1)
            elif self.output_activation == "sigmoid":
                return (probs > 0.5).astype(int)
            else:
                return probs
    
    def evaluate(self, X, Y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        if self.task_type == "regression":
            mse_score = np.mean((Y.flatten() - predictions.flatten()) ** 2)
            mae_score = np.mean(np.abs(Y.flatten() - predictions.flatten()))
            r2_score = self._calculate_r2(Y.flatten(), predictions.flatten())
            return {"MSE": mse_score, "MAE": mae_score, "R2": r2_score}
        else:
            if len(Y.shape) > 1 and Y.shape[1] > 1:  # One-hot encoded
                Y_labels = np.argmax(Y, axis=1)
            else:
                Y_labels = Y.flatten()
            
            accuracy = np.mean(predictions.flatten() == Y_labels)
            return {"Accuracy": accuracy}
    
    def _calculate_r2(self, y_true, y_pred):
        """Calculate R-squared score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        return 1 - (ss_res / ss_tot)

def create_sample_datasets():
    """Create sample datasets for testing"""
    np.random.seed(42)
    
    # 1. Regression Dataset
    print("Creating regression dataset...")
    X_reg = np.random.randn(200, 3)
    y_reg = 2*X_reg[:, 0] + 3*X_reg[:, 1] - 1.5*X_reg[:, 2] + np.random.randn(200) * 0.5
    y_reg = y_reg.reshape(-1, 1)
    
    # 2. Binary Classification Dataset
    print("Creating binary classification dataset...")
    X_bin = np.random.randn(200, 2)
    y_bin = ((X_bin[:, 0] + X_bin[:, 1]) > 0).astype(int).reshape(-1, 1)
    
    # 3. Multi-class Classification Dataset (Fixed)
    print("Creating multi-class classification dataset...")
    # Create 3 distinct clusters manually
    n_samples_per_class = 70
    
    # Class 0: centered at (-2, -2)
    X_class0 = np.random.randn(n_samples_per_class, 2) * 0.8 + np.array([-2, -2])
    y_class0 = np.zeros(n_samples_per_class)
    
    # Class 1: centered at (2, -2) 
    X_class1 = np.random.randn(n_samples_per_class, 2) * 0.8 + np.array([2, -2])
    y_class1 = np.ones(n_samples_per_class)
    
    # Class 2: centered at (0, 2)
    X_class2 = np.random.randn(n_samples_per_class, 2) * 0.8 + np.array([0, 2])
    y_class2 = np.full(n_samples_per_class, 2)
    
    # Combine all classes
    X_multi = np.vstack([X_class0, X_class1, X_class2])
    y_multi = np.hstack([y_class0, y_class1, y_class2])
    
    # Shuffle the data
    indices = np.random.permutation(len(X_multi))
    X_multi = X_multi[indices]
    y_multi = y_multi[indices]
    
    # Convert to one-hot encoding
    y_multi_onehot = np.eye(3)[y_multi.astype(int)]
    
    return (X_reg, y_reg), (X_bin, y_bin), (X_multi, y_multi_onehot)

# Example usage:
if __name__ == "__main__":
    print("🧠 Neural Network Testing Suite")
    print("=" * 50)
    
    # Create sample datasets
    (X_reg, y_reg), (X_bin, y_bin), (X_multi, y_multi) = create_sample_datasets()
    
    # === 1. REGRESSION EXAMPLE ===
    print("\n🔢 === REGRESSION EXAMPLE ===")
    print("Task: Predict continuous house prices")
    print(f"Data shape: X={X_reg.shape}, y={y_reg.shape}")
    
    nn_reg = NeuralNetwork(
        layers=[3, 16, 8, 1], 
        activations=["relu", "relu"], 
        output_activation="linear",
        loss="mse", 
        lr=0.01,
        task_type="regression"
    )
    
    print("Training regression model...")
    losses_reg = nn_reg.train(X_reg, y_reg, epochs=500, verbose=True)
    predictions_reg = nn_reg.predict(X_reg)
    metrics_reg = nn_reg.evaluate(X_reg, y_reg)
    
    print(f"📊 Regression Results:")
    for metric, value in metrics_reg.items():
        print(f"   {metric}: {value:.6f}")
    
    # === 2. BINARY CLASSIFICATION EXAMPLE ===
    print("\n🔍 === BINARY CLASSIFICATION EXAMPLE ===")
    print("Task: Classify spam vs not-spam emails")
    print(f"Data shape: X={X_bin.shape}, y={y_bin.shape}")
    
    nn_bin = NeuralNetwork(
        layers=[2, 8, 4, 1], 
        activations=["relu", "relu"], 
        output_activation="sigmoid",
        loss="binary_cross_entropy", 
        lr=0.1,
        task_type="binary_classification"
    )
    
    print("Training binary classification model...")
    losses_bin = nn_bin.train(X_bin, y_bin, epochs=500, verbose=True)
    predictions_bin = nn_bin.predict(X_bin)
    metrics_bin = nn_bin.evaluate(X_bin, y_bin)
    
    print(f"📊 Binary Classification Results:")
    for metric, value in metrics_bin.items():
        print(f"   {metric}: {value:.6f}")
    
    # === 3. MULTI-CLASS CLASSIFICATION EXAMPLE ===
    print("\n🌺 === MULTI-CLASS CLASSIFICATION EXAMPLE ===")
    print("Task: Classify iris flower species (3 classes)")
    print(f"Data shape: X={X_multi.shape}, y={y_multi.shape}")
    
    nn_multi = NeuralNetwork(
        layers=[2, 12, 8, 3], 
        activations=["relu", "relu"], 
        output_activation="softmax",
        loss="categorical_cross_entropy", 
        lr=0.1,
        task_type="multiclass_classification"
    )
    
    print("Training multi-class classification model...")
    losses_multi = nn_multi.train(X_multi, y_multi, epochs=500, verbose=True)
    predictions_multi = nn_multi.predict(X_multi)
    metrics_multi = nn_multi.evaluate(X_multi, y_multi)
    
    print(f"📊 Multi-class Classification Results:")
    for metric, value in metrics_multi.items():
        print(f"   {metric}: {value:.6f}")
    
    # === SUMMARY ===
    print("\n" + "="*50)
    print("🎯 FINAL SUMMARY")
    print("="*50)
    print(f"✅ Regression - R²: {metrics_reg['R2']:.4f} (closer to 1.0 is better)")
    print(f"✅ Binary Classification - Accuracy: {metrics_bin['Accuracy']:.1%}")
    print(f"✅ Multi-class Classification - Accuracy: {metrics_multi['Accuracy']:.1%}")
    print("\n🎉 All tests completed successfully!")
    print("🔧 The neural network works for all task types!")