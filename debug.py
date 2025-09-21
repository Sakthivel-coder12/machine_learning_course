
import tensorflow as tf
import os
import subprocess

print('=== DEBUG MODE ===')
print('TF Version:', tf.__version__)

# Check what CUDA version TF wants
build_info = tf.sysconfig.get_build_info()
print('Build info keys:', list(build_info.keys()))
if 'cuda_version' in build_info:
    print('TF wants CUDA:', build_info['cuda_version'])
else:
    print('NO CUDA VERSION IN BUILD INFO - This is CPU-only TF!')

# Check if CUDA paths exist
cuda_path = 'C:\\\\Program Files\\\\NVIDIA GPU Computing Toolkit\\\\CUDA\\\\v12.1\\\\bin'
print(f'\\\\nChecking CUDA path: {cuda_path}')
print('Path exists:', os.path.exists(cuda_path))

if os.path.exists(cuda_path):
    print('Files in CUDA bin:')
    for f in os.listdir(cuda_path):
        if 'cudart' in f or 'cudnn' in f:
            print(f'  - {f}')

# Check NVIDIA driver
print('\\\\n=== NVIDIA DRIVER INFO ===')
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    print('nvidia-smi output:')
    print(result.stdout[:500])  # First 500 chars
except Exception as e:
    print('nvidia-smi failed:', e)

print('\\\\n=== FINAL GPU CHECK ===')
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)
