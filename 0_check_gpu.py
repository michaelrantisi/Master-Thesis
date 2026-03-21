import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Capability:", torch.cuda.get_device_capability(0))
