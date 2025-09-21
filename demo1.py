import tensorflow as tf
import os

print('TF Version:', tf.__version__)
print('Build Info CUDA Version:', tf.sysconfig.get_build_info()['cuda_version'])
print('Build Info cuDNN Version:', tf.sysconfig.get_build_info()['cudnn_version'])

# Check conda environment CUDA path
cuda_path = os.path.join(os.environ['CONDA_PREFIX'], 'Library', 'bin')
print(f'\nLooking for CUDA DLLs in: {cuda_path}')

if os.path.exists(cuda_path):
    print('Files found:')
    cuda_files = [f for f in os.listdir(cuda_path) if 'cudart' in f or 'cudnn' in f or 'cublas' in f]
    for f in cuda_files:
        print(f'  - {f}')
else:
    print('CUDA path does not exist!')

print('\nTrying to list GPU devices...')
gpus = tf.config.list_physical_devices('GPU')
print('GPU devices:', gpus)
