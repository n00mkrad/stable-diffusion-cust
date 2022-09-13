import torch

for i in range(torch.cuda.device_count()):
    print(f"{i} - {torch.cuda.get_device_name(i)} - {round(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024, 2)} GB")