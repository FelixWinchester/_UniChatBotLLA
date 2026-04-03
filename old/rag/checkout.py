import torch
from transformers import __version__ as tf_version

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {tf_version}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")