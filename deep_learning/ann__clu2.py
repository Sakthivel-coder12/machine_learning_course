import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Activation Functions and Derivatives
def relu(z): 
    return np.maximum(0, z)

def relu_derivative(z): 
    return (z > 0).astype(float)

def sigmoid(z): 
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

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
    s = softmax(z)
    return s * (1 - s)

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

def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def mae_derivative(y_true, y_pred):
    return np.sign(y_pred - y_true) / y_true.shape[0]

# Registries
activation_functions = {
    "relu": (relu, relu_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative),
    "tanh": (tanh, tanh_derivative),
    "linear": (linear, linear_derivative),
    "softmax": (softmax, softmax_derivative)
}

loss_functions = {
    "mse": (mse, mse_derivative),
    "mae": (mae, mae_derivative),
    "binary_cross_entropy": (binary_cross_entropy, None),
    "categorical_cross_entropy": (categorical_cross_entropy, None)
}

class InteractiveNeuralNetwork:
    def __init__(self):
        self.layers = []
        self.activations = []
        self.output_activation = "linear"
        self.task_type = "regression"
        self.loss = "mse"
        self.lr = 0.01
        self.epochs = 1000
        self.params = {}
        self.cache = {}
        self.loss_fn = None
        self.loss_derivative = None
        
    def print_banner(self):
        print("=" * 60)
        print("🧠 INTERACTIVE NEURAL NETWORK BUILDER 🧠")
        print("=" * 60)
        print("Welcome! Let's build your neural network step by step.")
        print("I'll guide you through the process with helpful suggestions.\n")
    
    def get_task_type(self):
        print("📋 STEP 1: Choose your task type")
        print("-" * 30)
        
        while True:
            print("\nAvailable task types:")
            print("1. Regression (predicting continuous values)")
            print("2. Binary Classification (2 classes)")
            print("3. Multi-class Classification (3+ classes)")
            print("4. Use sample dataset for testing")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                self.task_type = "regression"
                print("✅ Great! You chose Regression.")
                break
            elif choice == "2":
                self.task_type = "binary_classification"
                print("✅ Great! You chose Binary Classification.")
                break
            elif choice == "3":
                self.task_type = "multiclass_classification"
                print("✅ Great! You chose Multi-class Classification.")
                break
            elif choice == "4":
                return self.get_sample_dataset()
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
    
    def get_sample_dataset(self):
        print("\n📊 Sample Dataset Options:")
        print("1. Regression dataset (house prices prediction)")
        print("2. Binary classification (spam detection)")
        print("3. Multi-class classification (iris flowers)")
        
        while True:
            choice = input("\nChoose sample dataset (1-3): ").strip()
            
            if choice == "1":
                self.task_type = "regression"
                X, y = make_regression(n_samples=500, n_features=8, noise=10, random_state=42)
                y = y.reshape(-1, 1)
                print("✅ Created regression dataset: 500 samples, 8 features")
                break
            elif choice == "2":
                self.task_type = "binary_classification"
                X, y = make_classification(n_samples=500, n_features=10, n_classes=2, 
                                         n_redundant=0, random_state=42)
                y = y.reshape(-1, 1)
                print("✅ Created binary classification dataset: 500 samples, 10 features")
                break
            elif choice == "3":
                self.task_type = "multiclass_classification"
                X, y = make_classification(n_samples=500, n_features=8, n_classes=3, 
                                         n_redundant=0, random_state=42)
                # Convert to one-hot
                y_one_hot = np.eye(3)[y]
                y = y_one_hot
                print("✅ Created multi-class classification dataset: 500 samples, 8 features, 3 classes")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        # Normalize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        print(f"📈 Dataset ready! Train: {self.X_train.shape[0]} samples, Test: {self.X_test.shape[0]} samples")
        
        # Set input size
        self.input_size = X.shape[1]
        return True
    
    def get_data_info(self):
        if hasattr(self, 'X_train'):
            return
            
        print("\n📊 STEP 2: Tell me about your data")
        print("-" * 35)
        
        while True:
            try:
                input_size = int(input("How many input features does your data have? "))
                if input_size > 0:
                    self.input_size = input_size
                    print(f"✅ Input size set to {input_size}")
                    break
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        if self.task_type == "multiclass_classification":
            while True:
                try:
                    num_classes = int(input("How many classes are you predicting? "))
                    if num_classes > 2:
                        self.num_classes = num_classes
                        print(f"✅ Number of classes set to {num_classes}")
                        break
                    else:
                        print("❌ Multi-class needs at least 3 classes.")
                except ValueError:
                    print("❌ Please enter a valid number.")
    
    def get_architecture(self):
        print("\n🏗️ STEP 3: Design your network architecture")
        print("-" * 42)
        
        # Start with input layer
        layers = [self.input_size]
        activations = []
        
        print(f"Input layer: {self.input_size} neurons ✅")
        
        # Get hidden layers
        print("\nNow let's add hidden layers:")
        layer_num = 1
        
        while True:
            print(f"\n--- Hidden Layer {layer_num} ---")
            
            # Suggest layer size
            if layer_num == 1:
                suggested = max(16, self.input_size // 2)
                print(f"💡 Suggestion: Start with {suggested} neurons")
            else:
                suggested = max(8, layers[-1] // 2)
                print(f"💡 Suggestion: Try {suggested} neurons (gradually decreasing)")
            
            neurons = input(f"Number of neurons (or 'done' to finish, 'auto' for {suggested}): ").strip()
            
            if neurons.lower() == 'done':
                if len(layers) == 1:
                    print("❌ You need at least one hidden layer!")
                    continue
                break
            elif neurons.lower() == 'auto':
                neurons = suggested
            else:
                try:
                    neurons = int(neurons)
                    if neurons <= 0:
                        print("❌ Please enter a positive number.")
                        continue
                except ValueError:
                    print("❌ Please enter a valid number, 'auto', or 'done'.")
                    continue
            
            layers.append(neurons)
            
            # Get activation for this layer
            activation = self.get_activation_for_layer(layer_num, "hidden")
            activations.append(activation)
            
            print(f"✅ Hidden layer {layer_num}: {neurons} neurons, {activation} activation")
            layer_num += 1
            
            if layer_num > 5:
                print("💡 That's quite a few layers! Consider finishing here.")
        
        # Add output layer
        output_size = self.get_output_size()
        layers.append(output_size)
        
        output_activation = self.get_output_activation()
        
        print(f"✅ Output layer: {output_size} neurons, {output_activation} activation")
        
        self.layers = layers
        self.activations = activations
        self.output_activation = output_activation
        
        print(f"\n🎯 Final Architecture: {' → '.join(map(str, layers))}")
        print(f"🎯 Activations: {' → '.join(activations + [output_activation])}")
    
    def get_activation_for_layer(self, layer_num, layer_type):
        print(f"\nChoose activation function for {layer_type} layer {layer_num}:")
        print("1. ReLU (most common, good for hidden layers)")
        print("2. Sigmoid (outputs 0-1)")
        print("3. Tanh (outputs -1 to 1)")
        print("4. Linear (no activation)")
        
        # Suggest based on layer type and task
        if layer_type == "hidden":
            suggestion = "1 (ReLU)"
        else:
            suggestion = "4 (Linear)"
        
        print(f"💡 Suggestion: {suggestion}")
        
        while True:
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == "1":
                return "relu"
            elif choice == "2":
                return "sigmoid"
            elif choice == "3":
                return "tanh"
            elif choice == "4":
                return "linear"
            else:
                print("❌ Please choose 1, 2, 3, or 4.")
    
    def get_output_size(self):
        if self.task_type == "regression":
            while True:
                try:
                    size = int(input("How many outputs are you predicting? (usually 1 for regression): "))
                    if size > 0:
                        return size
                    else:
                        print("❌ Please enter a positive number.")
                except ValueError:
                    print("❌ Please enter a valid number.")
        
        elif self.task_type == "binary_classification":
            return 1
        
        elif self.task_type == "multiclass_classification":
            if hasattr(self, 'num_classes'):
                return self.num_classes
            else:
                return 3  # Default for sample dataset
    
    def get_output_activation(self):
        print(f"\nChoose output activation for {self.task_type}:")
        
        if self.task_type == "regression":
            print("1. Linear (recommended for regression)")
            print("2. ReLU (if you know outputs are positive)")
            suggestion = "1"
            
            while True:
                choice = input(f"Enter choice (💡 suggestion: {suggestion}): ").strip()
                if choice == "1" or choice == "":
                    return "linear"
                elif choice == "2":
                    return "relu"
                else:
                    print("❌ Please choose 1 or 2.")
        
        elif self.task_type == "binary_classification":
            print("1. Sigmoid (recommended for binary classification)")
            return "sigmoid"
        
        elif self.task_type == "multiclass_classification":
            print("1. Softmax (recommended for multi-class)")
            return "softmax"
    
    def get_loss_function(self):
        print("\n📉 STEP 4: Choose loss function")
        print("-" * 30)
        
        if self.task_type == "regression":
            print("1. MSE (Mean Squared Error) - most common")
            print("2. MAE (Mean Absolute Error) - robust to outliers")
            
            while True:
                choice = input("Enter choice (💡 suggestion: 1): ").strip()
                if choice == "1" or choice == "":
                    self.loss = "mse"
                    print("✅ Using MSE loss")
                    break
                elif choice == "2":
                    self.loss = "mae"
                    print("✅ Using MAE loss")
                    break
                else:
                    print("❌ Please choose 1 or 2.")
        
        elif self.task_type == "binary_classification":
            self.loss = "binary_cross_entropy"
            print("✅ Using Binary Cross-Entropy loss (automatic)")
        
        elif self.task_type == "multiclass_classification":
            self.loss = "categorical_cross_entropy"
            print("✅ Using Categorical Cross-Entropy loss (automatic)")
    
    def get_training_params(self):
        print("\n⚙️ STEP 5: Set training parameters")
        print("-" * 33)
        
        # Learning rate
        print("Learning Rate controls how fast the network learns:")
        print("💡 0.001 = slow but stable, 0.01 = balanced, 0.1 = fast but risky")
        
        while True:
            lr_input = input("Enter learning rate (💡 suggestion: 0.01): ").strip()
            if lr_input == "":
                self.lr = 0.01
                break
            try:
                lr = float(lr_input)
                if 0 < lr <= 1:
                    self.lr = lr
                    break
                else:
                    print("❌ Please enter a value between 0 and 1.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        print(f"✅ Learning rate set to {self.lr}")
        
        # Epochs
        print(f"\nNumber of training epochs (iterations):")
        print("💡 500 = quick test, 1000 = good start, 2000+ = thorough training")
        
        while True:
            epochs_input = input("Enter number of epochs (💡 suggestion: 1000): ").strip()
            if epochs_input == "":
                self.epochs = 1000
                break
            try:
                epochs = int(epochs_input)
                if epochs > 0:
                    self.epochs = epochs
                    break
                else:
                    print("❌ Please enter a positive number.")
            except ValueError:
                print("❌ Please enter a valid number.")
        
        print(f"✅ Training epochs set to {self.epochs}")
    
    def display_summary(self):
        print("\n" + "="*50)
        print("📋 NETWORK SUMMARY")
        print("="*50)
        print(f"Task Type: {self.task_type}")
        print(f"Architecture: {' → '.join(map(str, self.layers))}")
        print(f"Activations: {' → '.join(self.activations + [self.output_activation])}")
        print(f"Loss Function: {self.loss}")
        print(f"Learning Rate: {self.lr}")
        print(f"Training Epochs: {self.epochs}")
        print("="*50)
        
        confirm = input("\n✅ Does this look good? (y/n): ").strip().lower()
        return confirm in ['y', 'yes', '']
    
    def initialize_network(self):
        """Initialize the neural network with collected parameters"""
        # Set loss function
        if self.loss in loss_functions:
            self.loss_fn, self.loss_derivative = loss_functions[self.loss]
        
        # Initialize weights
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            if i < len(self.activations) and self.activations[i] == "relu":
                std = np.sqrt(2.0 / self.layers[i])  # He initialization
            else:
                std = np.sqrt(1.0 / self.layers[i])  # Xavier initialization
            
            self.params[f"W{i+1}"] = np.random.randn(self.layers[i], self.layers[i+1]) * std
            self.params[f"b{i+1}"] = np.zeros((1, self.layers[i+1]))
        
        print("🔧 Network initialized successfully!")
    
    def forward(self, X):
        self.cache = {"A0": X}
        
        for i in range(1, len(self.layers)):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            Z = np.dot(self.cache[f"A{i-1}"], W) + b
            
            if i == len(self.layers) - 1:
                act_name = self.output_activation
            else:
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
        
        # Output layer gradient
        if (self.output_activation == "softmax" and self.loss == "categorical_cross_entropy") or \
           (self.output_activation == "sigmoid" and self.loss == "binary_cross_entropy"):
            dZ = A_final - Y
        else:
            if self.loss_derivative is not None:
                dA = self.loss_derivative(Y, A_final)
            else:
                dA = A_final - Y
            
            _, act_derivative = activation_functions[self.output_activation]
            dZ = dA * act_derivative(self.cache[f"Z{L}"])
        
        # Backpropagate
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f"A{i-1}"]
            W = self.params[f"W{i}"]
            
            grads[f"dW{i}"] = np.dot(A_prev.T, dZ) / m
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
            
            if i > 1:
                Z_prev = self.cache[f"Z{i-1}"]
                _, act_deriv = activation_functions[self.activations[i-2]]
                dA_prev = np.dot(dZ, W.T)
                dZ = dA_prev * act_deriv(Z_prev)
        
        # Update parameters
        for i in range(1, L + 1):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]
    
    def train(self):
        if not hasattr(self, 'X_train'):
            print("❌ No training data available. Please use sample dataset option.")
            return
        
        print("\n🚀 Starting training...")
        print("-" * 25)
        
        losses = []
        for epoch in range(self.epochs):
            Y_hat = self.forward(self.X_train)
            loss = self.loss_fn(self.y_train, Y_hat)
            losses.append(loss)
            self.backward(self.y_train)
            
            if (epoch + 1) % (self.epochs // 10) == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:4d}/{self.epochs}, Loss: {loss:.6f}")
        
        print("✅ Training completed!")
        
        # Plot training loss
        self.plot_training_loss(losses)
        
        # Evaluate on test set
        self.evaluate()
    
    def plot_training_loss(self, losses):
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('Training Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.show()
    
    def evaluate(self):
        if not hasattr(self, 'X_test'):
            return
        
        print("\n📊 EVALUATION RESULTS")
        print("-" * 25)
        
        # Make predictions
        test_pred = self.forward(self.X_test)
        
        if self.task_type == "regression":
            test_pred_values = test_pred
            mse_score = np.mean((self.y_test - test_pred_values) ** 2)
            mae_score = np.mean(np.abs(self.y_test - test_pred_values))
            
            print(f"Mean Squared Error: {mse_score:.6f}")
            print(f"Mean Absolute Error: {mae_score:.6f}")
            
            # R² score
            ss_res = np.sum((self.y_test - test_pred_values) ** 2)
            ss_tot = np.sum((self.y_test - np.mean(self.y_test)) ** 2)
            r2_score = 1 - (ss_res / ss_tot)
            print(f"R² Score: {r2_score:.6f}")
            
        else:
            if self.output_activation == "softmax":
                test_pred_labels = np.argmax(test_pred, axis=1)
                if len(self.y_test.shape) > 1:
                    y_true_labels = np.argmax(self.y_test, axis=1)
                else:
                    y_true_labels = self.y_test.flatten()
            else:
                test_pred_labels = (test_pred > 0.5).astype(int).flatten()
                y_true_labels = self.y_test.flatten()
            
            accuracy = np.mean(test_pred_labels == y_true_labels)
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    def run_interactive_session(self):
        """Main method to run the interactive session"""
        self.print_banner()
        
        try:
            # Step 1: Get task type and possibly sample data
            self.get_task_type()
            
            # Step 2: Get data info (if not using sample data)
            self.get_data_info()
            
            # Step 3: Design architecture
            self.get_architecture()
            
            # Step 4: Choose loss function
            self.get_loss_function()
            
            # Step 5: Set training parameters
            self.get_training_params()
            
            # Step 6: Review and confirm
            if not self.display_summary():
                print("Let's start over...")
                return self.run_interactive_session()
            
            # Step 7: Initialize and train
            self.initialize_network()
            
            if hasattr(self, 'X_train'):
                train_choice = input("\n🚀 Ready to train? (y/n): ").strip().lower()
                if train_choice in ['y', 'yes', '']:
                    self.train()
                else:
                    print("Network created but not trained. You can train it later!")
            else:
                print("✅ Network created successfully!")
                print("💡 To train, you'll need to provide training data (X_train, y_train)")
        
        except KeyboardInterrupt:
            print("\n\n👋 Exiting... Thanks for using Neural Network Builder!")
        except Exception as e:
            print(f"\n❌ An error occurred: {str(e)}")
            print("Please try again or contact support.")

# Main execution
if __name__ == "__main__":
    nn_builder = InteractiveNeuralNetwork()
    nn_builder.run_interactive_session()