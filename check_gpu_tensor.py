import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices('GPU'))
print("CUDA built:", tf.test.is_built_with_cuda())

# Detailed GPU info
if tf.config.list_physical_devices('GPU'):
    gpu = tf.config.list_physical_devices('GPU')[0]
    print("GPU details:", tf.config.experimental.get_device_details(gpu))
else:
    print("No GPU detected")