import tensorflow as tf
import sys
import subprocess

print("=" * 60)
print("TENSORFLOW INSTALLATION VERIFICATION")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"TensorFlow Version: {tf.__version__}")

# Check if CUDA is available
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# Check for GPU
gpu_devices = tf.config.list_physical_devices('GPU')
print(f"GPU devices found: {len(gpu_devices)}")

if gpu_devices:
    print("🎉 SUCCESS: GPU DETECTED!")
    for i, device in enumerate(gpu_devices):
        print(f"GPU {i}: {device}")
        
    # Test GPU operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 0.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f"Matrix multiplication result: {c.numpy()}")
        print(f"Operation executed on: {c.device}")
else:
    print("😥 No GPU devices found")
    print("\nBut let's at least verify TensorFlow works on CPU:")
    # Test basic functionality
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    c = a + b
    print(f"Basic operation: {a} + {b} = {c}")
    print("✓ TensorFlow is working (on CPU)!")

print("\n" + "=" * 60)